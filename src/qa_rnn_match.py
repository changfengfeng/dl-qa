# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import time
import word2vec as w2v
from qa_data import TokenLoader
from metrics import Evaluator

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("word_vec_path", "data/word_vec.txt", "")
tf.flags.DEFINE_string("train_data_path", "data/qa/train_id.data", "")
tf.flags.DEFINE_string("validate_data_path", "data/qa/validate_id.data", "")
tf.flags.DEFINE_integer("max_sequence_length", 150, "")
tf.flags.DEFINE_integer("hidden_units", 100, "")
tf.flags.DEFINE_float("learning_rate", 0.001, "")
tf.flags.DEFINE_integer("batch_size", 100, "")
tf.flags.DEFINE_string("log_dir", "log", "")

class Model:
    """ Using the conv + max pooling to do sequence match
    """

    def __init__(self, w2v_embedding, max_sequence_length, hidden_units,
            learning_rate):

        self.max_sequence_length = max_sequence_length
        self.learning_rate = learning_rate
        self.embedding_size = w2v_embedding.shape[1]
        self.hidden_units = hidden_units

        with tf.Graph().as_default() as graph:
            self.question_holder = tf.placeholder(tf.int32, [None,
                max_sequence_length], name="question")
            self.answer_holder = tf.placeholder(tf.int32, [None,
                max_sequence_length], name="answer")

            self.embedding = tf.Variable(w2v_embedding, dtype=tf.float32,
                name="embedding")
            self.sim_matrix = tf.get_variable(name="sim_matrix",
                    shape=[hidden_units * 2, hidden_units * 2], dtype=tf.float32,
                regularizer=tf.contrib.layers.l2_regularizer(0.01),
                initializer=tf.contrib.layers.xavier_initializer())

            self.project_weight = tf.get_variable(name="projection_weight",
                shape=[4 * hidden_units + 1, 2], dtype=tf.float32,
                regularizer=tf.contrib.layers.l2_regularizer(0.001),
                initializer=tf.contrib.layers.xavier_initializer())
            self.project_biase = tf.Variable(tf.zeros([2]), dtype=tf.float32,
                name="projection_bias")

            self.graph = graph

    def inference(self, question, answer, reuse):
        question_input = tf.nn.embedding_lookup(self.embedding, question)
        question_length = tf.reduce_sum(tf.sign(question), 1)

        answer_input = tf.nn.embedding_lookup(self.embedding, answer)
        answer_length = tf.reduce_sum(tf.sign(answer), 1)

        with tf.variable_scope("q-bilstm", reuse=reuse) as scope:
            forward_output, _ = tf.nn.dynamic_rnn(
                    tf.nn.rnn_cell.LSTMCell(self.hidden_units, reuse=reuse),
                    question_input,
                    sequence_length = question_length,
                    dtype=tf.float32, scope="forward_rnn")
            backward_input = tf.reverse_sequence(question_input,
                    seq_lengths=question_length, seq_dim=1)
            backward_output, _ = tf.nn.dynamic_rnn(
                    tf.nn.rnn_cell.LSTMCell(self.hidden_units, reuse=reuse),
                    backward_input,
                    sequence_length = question_length,
                    dtype=tf.float32, scope="backward_rnn")
        question_backward_output = tf.reverse_sequence(backward_output,
                seq_lengths=question_length, seq_dim=1)

        question_output = tf.concat([forward_output, question_backward_output],
                2)
        question_output = tf.reduce_mean(question_output, 1)

        with tf.variable_scope("a-bilstm", reuse=reuse) as scope:
            forward_output, _ = tf.nn.dynamic_rnn(
                    tf.nn.rnn_cell.LSTMCell(self.hidden_units, reuse=reuse),
                    answer_input,
                    sequence_length = answer_length,
                    dtype=tf.float32, scope="forward_rnn")
            backward_input = tf.reverse_sequence(answer_input,
                    seq_lengths=answer_length, seq_dim=1)
            backward_output, _ = tf.nn.dynamic_rnn(
                    tf.nn.rnn_cell.LSTMCell(self.hidden_units, reuse=reuse),
                    backward_input,
                    sequence_length = answer_length,
                    dtype=tf.float32, scope="backward_rnn")
        answer_backward_output = tf.reverse_sequence(backward_output,
                seq_lengths=answer_length, seq_dim=1)

        answer_output = tf.concat([forward_output, answer_backward_output],
                2)
        answer_output = tf.reduce_mean(answer_output, 1)

        if not reuse:
            question_output = tf.nn.dropout(question_output, 0.5)
            answer_output = tf.nn.dropout(answer_output, 0.5)

        with tf.variable_scope("sim", reuse=reuse):
            qsim = tf.matmul(question_output, self.sim_matrix)
            asim = tf.matmul(qsim, tf.transpose(answer_output, [1, 0]))
            sim_output = tf.diag_part(asim)
            sim_output = tf.expand_dims(sim_output, -1)
        print(question_output, sim_output, answer_output)
        sentence_output = tf.concat([question_output, sim_output,
            answer_output], -1)

        logits = tf.add(tf.matmul(sentence_output, self.project_weight),
                self.project_biase, name="logits")

        return logits

    def loss(self, question, answer, label, weight):
        logits_val = self.inference(question, answer, False)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=label,
                logits=logits_val, weights=weight)
        return tf.reduce_mean(loss)

    def train_op(self, loss):
        return tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

    def validate_op(self):
        logits = self.inference(self.question_holder, self.answer_holder, True)
        return tf.nn.softmax(logits)

    def validate_accuray(self, sess, softmax, question_input, answer_input,
            label_input):
        batch_size = 100
        batches = len(question_input) // batch_size

        correct = 0
        for batch in range(batches):
            questions = question_input[batch * batch_size:(batch + 1) *
                    batch_size,:]
            answers = answer_input[batch * batch_size: (batch + 1) * batch_size,:]
            labels = label_input[batch * batch_size: (batch + 1) * batch_size]
            softmax_val = sess.run(softmax,
                    {self.question_holder: questions, self.answer_holder:
                        answers})
            predicted = np.argmax(softmax_val, 1)
            correct += np.sum(np.equal(predicted, labels))

        if batch * batch_size < len(question_input):
            questions = question_input[batch * batch_size:,:]
            answers = answer_input[batch * batch_size:,:]
            labels = label_input[batch * batch_size:]
            softmax_val = sess.run(softmax,
                    {self.question_holder: questions, self.answer_holder:
                        answers})
            predicted = np.argmax(softmax_val, 1)
            correct += np.sum(np.equal(predicted, labels))

        print("accuray: {:.4f}".format(correct / len(question_input)))

    def validate(self, sess, softmax, question_input, answer_input, label_input):
        # groupy question and answer by same question
        batch_size = 100
        batches = len(question_input) // batch_size

        qa_pairs = []
        scores = []

        for batch in range(batches):
            questions = question_input[batch * batch_size:(batch + 1) *
                    batch_size,:]
            answers = answer_input[batch * batch_size: (batch + 1) * batch_size,:]
            labels = label_input[batch * batch_size: (batch + 1) * batch_size]
            softmax_val = sess.run(softmax,
                    {self.question_holder: questions, self.answer_holder:
                        answers})
            for q, a, l, score in zip(questions, answers, labels, softmax_val):
                qa_pairs.append([np.array(q).astype(str),
                    np.array(a).astype(str), l])
                scores.append(score[1])

                if batch == 0:
                    print(q[:2], l, score)

        if batches * batch_size < len(question_input):
            questions = question_input[batches * batch_size:,:]
            answers = answer_input[batches * batch_size:,:]
            labels = label_input[batches * batch_size:]
            softmax_val = sess.run(softmax,
                    {self.question_holder: questions, self.answer_holder:
                        answers})

            for q, a, l, score in zip(questions, answers, labels, softmax_val):
                qa_pairs.append([np.array(q).astype(str),
                    np.array(a).astype(str), l])
                scores.append(score[1])

        evaluator = Evaluator(qa_pairs, scores)
        evaluator.calculate()

        print("map {:.4f}  mrr {:.4f} acc@1 {:.4f}".format(
            evaluator.MAP(), evaluator.MRR(), evaluator.ACC_at_1()))

    def train(self, batch_size, train_data_path, validate_data_path):
        token_loader = TokenLoader(self.max_sequence_length)
        # load data tokens
        train_questions, train_answers, train_labels = token_loader.load_tokens(train_data_path)
        validate_questions, validate_answers, validate_labels = token_loader.load_tokens(validate_data_path)

        batches = len(train_questions) // batch_size

        with self.graph.as_default():
            question_tensor = tf.placeholder(tf.int32, [batch_size,
                self.max_sequence_length])
            answer_tensor = tf.placeholder(tf.int32, [batch_size,
                self.max_sequence_length])
            label_tensor = tf.placeholder(tf.int32, [batch_size])
            weight_tensor = tf.placeholder(tf.float32, [batch_size])
            loss = self.loss(question_tensor, answer_tensor, label_tensor,
                weight_tensor)
            train_op = self.train_op(loss)

            softmax = self.validate_op()

            sv = tf.train.Supervisor(graph=self.graph, logdir=FLAGS.log_dir)

        with sv.managed_session(master='') as sess:
            step = 0
            for epoch in range(10):
                for batch in range(batches):
                    questions = train_questions[batch * batch_size: (batch + 1) *
                            batch_size,:]
                    answers = train_answers[batch * batch_size: (batch + 1) *
                            batch_size,:]
                    labels = train_labels[batch * batch_size: (batch + 1) *
                            batch_size]
                    num_pos = np.sum(labels)
                    pos_weight = float(batch_size) / num_pos - 2 if num_pos > 0 else 0.0
                    weights = np.ones(batch_size, "float32")
                    if num_pos > 0:
                        weights += labels * pos_weight

                    loss_val, _ = sess.run([loss, train_op],
                            {question_tensor: questions,
                             answer_tensor: answers,
                             label_tensor: labels,
                             weight_tensor: weights})

                    if (step + 1) % 10 == 0:
                        print("loss {:.4f} step {}".format(loss_val, step + 1))

                    if step == 0 or (step + 1) % 1000 == 0:
                        #self.validate_accuray(sess, softmax, validate_questions,
                        #        validate_answers, validate_labels)
                        self.validate(sess, softmax, validate_questions,
                                validate_answers, validate_labels)

def main(_):
    w2v_model = w2v.load(FLAGS.word_vec_path)
    #vectors = load_w2v(FLAGS.word_vec_path)
    model = Model(w2v_model.vectors, FLAGS.max_sequence_length,
            FLAGS.hidden_units, FLAGS.learning_rate)
    model.train(FLAGS.batch_size, FLAGS.train_data_path,
            FLAGS.validate_data_path)

if __name__ == "__main__":
    tf.app.run()
