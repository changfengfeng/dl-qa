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
tf.flags.DEFINE_integer("filter_size", 5, "")
tf.flags.DEFINE_integer("num_filter", 100, "")
tf.flags.DEFINE_float("learning_rate", 0.0001, "")
tf.flags.DEFINE_integer("batch_size", 128, "")
tf.flags.DEFINE_string("log_dir", "log", "")

class Model:
    """ Using the conv + max pooling to do sequence match
    """

    def __init__(self, w2v_embedding, max_sequence_length, filter_size, num_filter,
            learning_rate):

        self.max_sequence_length = max_sequence_length
        self.filter_size = filter_size
        self.num_filter = num_filter
        self.learning_rate = learning_rate
        self.embedding_size = w2v_embedding.shape[1]

        with tf.Graph().as_default() as graph:
            self.question_holder = tf.placeholder(tf.int32, [None,
                max_sequence_length], name="question")
            self.answer_holder = tf.placeholder(tf.int32, [None,
                max_sequence_length], name="answer")

            self.filter = tf.get_variable(name="filter", shape=[filter_size,
                self.embedding_size, 1, num_filter], dtype=tf.float32,
                regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                initializer=tf.contrib.layers.xavier_initializer())

            self.embedding = tf.Variable(w2v_embedding, dtype=tf.float32,
                name="embedding")
            self.sim_matrix = tf.get_variable(name="sim_matrix", shape=[num_filter,
                num_filter], dtype=tf.float32,
                regularizer=tf.contrib.layers.l2_regularizer(0.01),
                initializer=tf.contrib.layers.xavier_initializer())

            self.project_weight = tf.get_variable(name="projection_weight",
                shape=[2 * num_filter + 1, 2], dtype=tf.float32,
                regularizer=tf.contrib.layers.l2_regularizer(0.001),
                initializer=tf.contrib.layers.xavier_initializer())
            self.project_biase = tf.Variable(tf.zeros([2]), dtype=tf.float32,
                name="projection_bias")

            self.graph = graph

    def inference(self, question, answer, reuse):
        question_input = tf.nn.embedding_lookup(self.embedding, question)
        question_input = tf.expand_dims(question_input, -1)
        answer_input = tf.nn.embedding_lookup(self.embedding, answer)
        answer_input = tf.expand_dims(answer_input, -1)

        with tf.variable_scope("conv", reuse=reuse) as scope:
            question_conv = tf.nn.conv2d(question_input, self.filter, strides=[1,
               1, 1, 1], padding="VALID")
            question_conv = tf.nn.relu(question_conv)
            question_pooling = tf.nn.max_pool(question_conv, ksize=[1,
                self.max_sequence_length - self.filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding="VALID")
            question_output = tf.squeeze(question_pooling, [1, 2])

            scope.reuse_variables()

            answer_conv = tf.nn.conv2d(answer_input, self.filter, strides=[1,
               1, 1, 1], padding="VALID")
            answer_conv = tf.nn.relu(answer_conv)
            answer_pooling = tf.nn.max_pool(answer_conv, ksize=[1,
                self.max_sequence_length - self.filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding="VALID")
            answer_output = tf.squeeze(answer_pooling, [1, 2])

        with tf.variable_scope("sim", reuse=reuse):
            qsim = tf.matmul(question_output, self.sim_matrix)
            asim = tf.matmul(qsim, tf.transpose(answer_output, [1, 0]))
            sim_output = tf.diag_part(asim)
            sim_output = tf.expand_dims(sim_output, -1)

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

        if batch * batch_size < len(question_input):
            questions = question_input[batch * batch_size:,:]
            answers = answer_input[batch * batch_size:,:]
            labels = label_input[batch * batch_size:]
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
                        self.validate(sess, softmax, validate_questions,
                                validate_answers, validate_labels)
                    step += 1

def load_w2v(path):
    fp = open(path, "r")
    print("load data from:", path)
    line = fp.readline().strip()
    ss = line.split(" ")
    total = int(ss[0])
    dim = int(ss[1])
    ws = []
    mv = [0 for i in range(dim)]
    for t in range(total):
        line = fp.readline().strip()
        ss = line.split(" ")
        assert (len(ss) == (dim + 1))
        vals = []
        for i in range(1, dim + 1):
            fv = float(ss[i])
            mv[i - 1] += fv
            vals.append(fv)
        ws.append(vals)
    for i in range(dim):
        mv[i] = mv[i] / total
    ws.append(mv)
    fp.close()
    return np.asarray(ws, dtype=np.float32)

def main(_):
    w2v_model = w2v.load(FLAGS.word_vec_path)
    #vectors = load_w2v(FLAGS.word_vec_path)
    model = Model(w2v_model.vectors, FLAGS.max_sequence_length,
            FLAGS.filter_size, FLAGS.num_filter, FLAGS.learning_rate)
    model.train(FLAGS.batch_size, FLAGS.train_data_path,
            FLAGS.validate_data_path)

if __name__ == "__main__":
    tf.app.run()
