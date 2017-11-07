# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import time
import word2vec as w2v
from insurance_data import TokenLoader
from metrics import Evaluator

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("word_vec_path", "data/word_vec.txt", "")
tf.flags.DEFINE_string("train_data_path", "data/insurance_train", "")
tf.flags.DEFINE_string("validate_data_path", "data/insurance_valid", "")
tf.flags.DEFINE_integer("max_sequence_length", 200, "")
tf.flags.DEFINE_integer("hidden_units", 100, "")
tf.flags.DEFINE_float("learning_rate", 0.1, "")
tf.flags.DEFINE_integer("batch_size", 100, "")
tf.flags.DEFINE_string("log_dir", "log", "")
tf.flags.DEFINE_float("margin", 0.1, "")

class Model:
    """ Using the conv + max pooling to do sequence match
    """

    def __init__(self, w2v_embedding, max_sequence_length, hidden_units,
            learning_rate, margin):

        self.max_sequence_length = max_sequence_length
        self.learning_rate = learning_rate
        self.embedding_size = w2v_embedding.shape[1]
        self.hidden_units = hidden_units
        self.margin = margin

        with tf.Graph().as_default() as graph:
            self.question_holder = tf.placeholder(tf.int32, [None,
                max_sequence_length], name="question")
            self.answer_holder = tf.placeholder(tf.int32, [None,
                max_sequence_length], name="answer")
            self.wrong_answer_holder = tf.placeholder(tf.int32, [None,
                max_sequence_length], name="wrong_answer")

            self.embedding = tf.Variable(w2v_embedding, dtype=tf.float32,
                name="embedding")

            self.graph = graph

    def bilstm(self, sequence, sequence_length, reuse):
        with tf.variable_scope("bilstm", reuse=reuse):
            forward_output, _ = tf.nn.dynamic_rnn(
                    tf.nn.rnn_cell.LSTMCell(self.hidden_units, reuse=reuse),
                    sequence,
                    sequence_length = sequence_length,
                    dtype=tf.float32, scope="forward_rnn")
            backward_input = tf.reverse_sequence(sequence,
                    seq_lengths=sequence_length, seq_dim=1)
            backward_output, _ = tf.nn.dynamic_rnn(
                    tf.nn.rnn_cell.LSTMCell(self.hidden_units, reuse=reuse),
                    backward_input,
                    sequence_length = sequence_length,
                    dtype=tf.float32, scope="backward_rnn")
        backward_output = tf.reverse_sequence(backward_output,
                seq_lengths=sequence_length, seq_dim=1)

        output = tf.concat([forward_output, backward_output],
                2)
        #output = tf.reduce_max(output, 1)
        output = tf.expand_dims(output, -1)
        output = tf.nn.max_pool(output, ksize=[1, self.max_sequence_length, 1,
            1], strides=[1,1,1,1], padding='VALID')
        output = tf.squeeze(output, [1, 3])
        return output

    def cos_sim(self, q, a):
        q1 = tf.sqrt(tf.reduce_sum(tf.multiply(q, q), 1))
        a1 = tf.sqrt(tf.reduce_sum(tf.multiply(a, a), 1))
        mul = tf.reduce_sum(tf.multiply(q, a), 1)
        cos_sim = tf.div(mul, tf.multiply(q1, a1))
        return cos_sim

    def inference(self, question, answer, wrong_answer=None, reuse=False):
        question_input = tf.nn.embedding_lookup(self.embedding, question)
        question_length = tf.reduce_sum(tf.sign(question), 1)

        answer_input = tf.nn.embedding_lookup(self.embedding, answer)
        answer_length = tf.reduce_sum(tf.sign(answer), 1)

        question_output = self.bilstm(question_input, question_length, reuse)
        answer_output = self.bilstm(answer_input, answer_length, True)

        if wrong_answer != None:
            wrong_answer_input = tf.nn.embedding_lookup(self.embedding,
                    wrong_answer)
            wrong_answer_length = tf.reduce_sum(tf.sign(wrong_answer), 1)
            wrong_answer_output = self.bilstm(wrong_answer_input,
                    wrong_answer_length, True)

        if not reuse:
            question_output = tf.nn.dropout(question_output, 0.5)
            answer_output = tf.nn.dropout(answer_output, 0.5)
            if wrong_answer != None:
                wrong_answer_output = tf.nn.dropout(wrong_answer_output, 0.5)

        # calc sim
        right_sim = self.cos_sim(question_output, answer_output)
        if wrong_answer != None:
            wrong_sim = self.cos_sim(question_output, wrong_answer_output)
            return right_sim, wrong_sim
        return right_sim

    def loss(self, question, answer, wrong_answer):
        right_sim, wrong_sim = self.inference(question, answer, wrong_answer, False)
        zero = tf.fill(tf.shape(right_sim), 0.0)
        margin = tf.fill(tf.shape(right_sim), self.margin)
        with tf.name_scope("loss"):
            losses = tf.maximum(zero, tf.subtract(margin, tf.subtract(right_sim,
                wrong_sim)))
            loss = tf.reduce_sum(losses)
        return loss

    def train_op(self, loss):
        return tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

    def validate_op(self):
        right_sim = self.inference(self.question_holder, self.answer_holder,
                reuse=True)
        return right_sim

    def validate(self, sess, sim, question_input, answer_input, label_input):
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
            sim_val = sess.run(sim,
                    {self.question_holder: questions, self.answer_holder:
                        answers})
            for q, a, l, score in zip(questions, answers, labels, sim_val):
                qa_pairs.append([np.array(q).astype(str),
                    np.array(a).astype(str), l])
                scores.append(score)

                if batch == 0:
                    print(q[:2], l, score, a[:2])


        if batch * batch_size < len(question_input):
            questions = question_input[batch * batch_size:,:]
            answers = answer_input[batch * batch_size:,:]
            labels = label_input[batch * batch_size:]
            sim_val = sess.run(sim,
                    {self.question_holder: questions, self.answer_holder:
                        answers})

            for q, a, l, score in zip(questions, answers, labels, sim_val):
                qa_pairs.append([np.array(q).astype(str),
                    np.array(a).astype(str), l])
                scores.append(score)

        evaluator = Evaluator(qa_pairs, scores)
        evaluator.calculate()

        print("map {:.4f}  mrr {:.4f} acc@1 {:.4f}".format(
            evaluator.MAP(), evaluator.MRR(), evaluator.ACC_at_1()))

    def train(self, batch_size, train_data_path, validate_data_path):
        token_loader = TokenLoader(self.max_sequence_length)
        # load data tokens
        train_questions, train_answers, train_wrong_answers = token_loader.load_true_false(train_data_path)
        print(train_questions.shape, train_answers.shape,
                train_wrong_answers.shape)
        validate_questions, validate_answers, validate_labels = token_loader.load_tokens(validate_data_path)

        batches = len(train_questions) // batch_size

        with self.graph.as_default():
            question_tensor = tf.placeholder(tf.int32, [batch_size,
                self.max_sequence_length])
            answer_tensor = tf.placeholder(tf.int32, [batch_size,
                self.max_sequence_length])
            wrong_answer_tensor = tf.placeholder(tf.int32, [batch_size,
                self.max_sequence_length])

            loss = self.loss(question_tensor, answer_tensor, wrong_answer_tensor)
            train_op = self.train_op(loss)
            sim = self.validate_op()

            sv = tf.train.Supervisor(graph=self.graph, logdir=FLAGS.log_dir)

        with sv.managed_session(master='') as sess:
            step = 0
            for epoch in range(10):
                for batch in range(batches):
                    questions = train_questions[batch * batch_size: (batch + 1) *
                            batch_size,:]
                    answers = train_answers[batch * batch_size: (batch + 1) *
                            batch_size,:]
                    wrong_answers = train_wrong_answers[batch * batch_size: (batch + 1) *
                            batch_size,:]

                    loss_val, _ = sess.run([loss, train_op],
                            {question_tensor: questions,
                             answer_tensor: answers,
                             wrong_answer_tensor: wrong_answers})

                    if (step + 1) % 10 == 0:
                        print("loss {:.4f} step {}".format(loss_val, step + 1))

                    if step == 0 or (step + 1) % 500 == 0:
                        self.validate(sess, sim, validate_questions,
                                validate_answers, validate_labels)
                    step += 1

def main(_):
    w2v_model = w2v.load(FLAGS.word_vec_path)
    #vectors = load_w2v(FLAGS.word_vec_path)
    model = Model(w2v_model.vectors, FLAGS.max_sequence_length,
            FLAGS.hidden_units, FLAGS.learning_rate, FLAGS.margin)
    model.train(FLAGS.batch_size, FLAGS.train_data_path,
            FLAGS.validate_data_path)

if __name__ == "__main__":
    tf.app.run()
