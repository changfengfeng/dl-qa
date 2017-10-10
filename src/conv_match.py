# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import time
import word2vec as w2v
from qa_data import TokenLoader

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("word_vec_path", "data/word_vec.txt", "")
tf.flags.DEFINE_string("train_data_path", "data/qa/train_id.data", "")
tf.flags.DEFINE_string("validate_data_path", "data/qa/validate_id.data", "")
tf.flags.DEFINE_integer("max_sequence_length", 80, "")
tf.flags.DEFINE_integer("filter_size", 5, "")
tf.flags.DEFINE_integer("num_filter", 100, "")
tf.flags.DEFINE_float("learning_rate", 0.001, "")
tf.flags.DEFINE_integer("batch_size", 64, "")

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

        self.question_holder = tf.placeholder(tf.int32, [None,
            max_sequence_length], name="question")
        self.answer_holder = tf.placeholder(tf.int32, [None,
            max_sequence_length], name="answer")

        # TODO regualizer
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

    def loss(self, question, answer, label):
        logits = self.inference(question, answer, False)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label,
                logits=logits, name="loss")
        return tf.reduce_mean(loss)

    def train_op(self, loss):
        return tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

    def validate_op(self):
        logits = self.inference(self.question_holder, self.answer_holder, True)
        return logits

    def validate(self, logits, question_input, answer_input, labels):
        pass

    def train(self, batch_size, train_data_path, validate_data_path):
        token_loader = TokenLoader(self.max_sequence_length)
        # load data tokens
        train_questions, train_answers, train_labels = token_loader.load_tokens(train_data_path)
        validate_questions, validate_answers, validate_labels = token_loader.load_tokens(validate_data_path)

        question_tensor = tf.placeholder(tf.int32, [batch_size,
            self.max_sequence_length])
        answer_tensor = tf.placeholder(tf.int32, [batch_size,
            self.max_sequence_length])
        label_tensor = tf.placeholder(tf.int32, [batch_size])
        loss = self.loss(question_tensor, answer_tensor, label_tensor)
        train_op = self.train_op(loss)

        batches = len(train_questions) // batch_size

        with tf.Session() as sess:
            step = 0
            for epoch in range(10):
                for batch in range(batches):
                    questions = train_questions[batch * batch_size: (batch + 1) *
                            batch_size,:]
                    answers = train_answers[batch * batch_size: (batch + 1) *
                            batch_size,:]
                    labels = train_answers[batch * batch_size: (batch + 1) *
                            batch_size]

                    loss, _ = sess.run([loss, train_op],
                            {question_tensor: questions,
                             answer_tensor: answers,
                             label_tensor: labels})

                    print("loss {:.4f} step {}".format(loss, step + 1))
                    step += 1

def main(_):
    w2v_model = w2v.load(FLAGS.word_vec_path)
    model = Model(w2v_model.vectors, FLAGS.max_sequence_length,
            FLAGS.filter_size, FLAGS.num_filter, FLAGS.learning_rate)
    model.train(FLAGS.batch_size, FLAGS.train_data_path,
            FLAGS.validate_data_path)

if __name__ == "__main__":
    tf.app.run()
