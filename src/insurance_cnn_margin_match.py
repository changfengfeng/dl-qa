# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import time
import word2vec as w2v
from insurance_data import TokenLoader
from metrics import Evaluator

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("word_vec_path", "data/word_vec_insurance.txt", "")
tf.flags.DEFINE_string("answer_input", "data/insurance_answer", "")

tf.flags.DEFINE_string("train_question_id_fn",
        "data/insurance_train_question_id_sample", "")
tf.flags.DEFINE_string("train_question_id_eval_fn",
        "data/insurance_train_question_id_eval", "")

tf.flags.DEFINE_string("train_question_token_fn", "data/insurance_train_question", "")
tf.flags.DEFINE_string("validate_question_token_fn",
        "data/insurance_valid_question", "")

tf.flags.DEFINE_string("train_json_fn", "data/pool/train.json.gz", "")
tf.flags.DEFINE_string("validate_json_fn", "data/pool/valid.json.gz", "")

tf.flags.DEFINE_integer("max_sequence_length", 200, "")
tf.flags.DEFINE_integer("filter_size", 5, "")
tf.flags.DEFINE_integer("num_filter", 100, "")
tf.flags.DEFINE_float("learning_rate", 0.0001, "")
tf.flags.DEFINE_integer("batch_size", 100, "")
tf.flags.DEFINE_string("log_dir", "log", "")
tf.flags.DEFINE_float("margin", 0.2, "")
tf.flags.DEFINE_integer("max_train_step", 200000, "")

class Model:
    """ Using the conv + max pooling to do sequence match
    """

    def __init__(self, w2v_embedding, max_sequence_length, filter_size, num_filter,
            learning_rate, margin, max_train_step):

        self.max_sequence_length = max_sequence_length
        self.filter_size = filter_size
        self.num_filter = num_filter
        self.learning_rate = learning_rate
        self.embedding_size = w2v_embedding.shape[1]
        self.margin = margin
        self.max_train_step = max_train_step

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

            self.graph = graph

    def inference(self, question, answer, false_answer, reuse):
        question_input = tf.nn.embedding_lookup(self.embedding, question)
        question_input = tf.expand_dims(question_input, -1)
        answer_input = tf.nn.embedding_lookup(self.embedding, answer)
        answer_input = tf.expand_dims(answer_input, -1)

        if false_answer != None:
            false_answer_input = tf.nn.embedding_lookup(self.embedding,
                    false_answer)
            false_answer_input = tf.expand_dims(false_answer_input, -1)

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

            if false_answer != None:
                scope.reuse_variables()
                false_answer_conv = tf.nn.conv2d(false_answer_input, self.filter,
                        strides=[1, 1, 1, 1], padding="VALID")
                false_answer_conv = tf.nn.relu(false_answer_conv)
                false_answer_pooling = tf.nn.max_pool(false_answer_conv, ksize=[1,
                    self.max_sequence_length - self.filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding="VALID")
                false_answer_output = tf.squeeze(false_answer_pooling, [1, 2])

        with tf.variable_scope("sim", reuse=reuse):
            true_sim = self.cos_sim(question_output, answer_output)
            if false_answer != None:
                false_sim = self.cos_sim(question_output, false_answer_output)
                if not reuse:
                    self.true_sim = true_sim
                    self.false_sim = false_sim
                return true_sim, false_sim
            return true_sim

    def cos_sim(self, q, a):
        q1 = tf.sqrt(tf.reduce_sum(tf.multiply(q, q), 1))
        a1 = tf.sqrt(tf.reduce_sum(tf.multiply(a, a), 1))
        mul = tf.reduce_sum(tf.multiply(q, a), 1)
        cos_sim = tf.div(mul, tf.multiply(q1, a1))
        return cos_sim

    def loss(self, question, answer, false_answer):
        right_sim, wrong_sim = self.inference(question, answer, false_answer, False)
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
        true_sim = self.inference(self.question_holder, self.answer_holder, None, True)
        return true_sim

    def validate(self, sess, softmax, question_input, answer_input, label_input):
        # groupy question and answer by same question
        batch_size = 100
        batches = len(question_input) // batch_size

        qa_pairs = []
        scores = []

        for batch in range(batches):
            if batch == 0 or batch % 1000 == 0:
                print("validate @", batch)
            questions = question_input[batch * batch_size:(batch + 1) *
                    batch_size,:]
            answers = answer_input[batch * batch_size: (batch + 1) * batch_size,:]
            labels = label_input[batch * batch_size: (batch + 1) * batch_size]
            softmax_val = sess.run(softmax,
                    {self.question_holder: questions, self.answer_holder:
                        answers})
            #print(questions)
            #print(answers)
            #print(softmax_val)
            meet_true = False
            for q, a, l, score in zip(questions, answers, labels, softmax_val):
                qa_pairs.append([np.array(q).astype(str),
                    np.array(a).astype(str), l])
                scores.append(score)
                if l == 1:
                    meet_true = True
                    #print(q[:10], a[:10], l, score)
                else:
                    if meet_true:
                        #print(q[:10], a[:10], l, score)
                        meet_true = False


        if batches * batch_size < len(question_input):
            questions = question_input[batches * batch_size:,:]
            answers = answer_input[batches * batch_size:,:]
            labels = label_input[batches * batch_size:]
            softmax_val = sess.run(softmax,
                    {self.question_holder: questions, self.answer_holder:
                        answers})
            #print(questions)
            #print(answers)
            #print(softmax_val)
            meet_true = False
            for q, a, l, score in zip(questions, answers, labels, softmax_val):
                qa_pairs.append([np.array(q).astype(str),
                    np.array(a).astype(str), l])
                scores.append(score)
                if l == 1:
                    meet_true = True
                    #print(q[:10], a[:10], l, score)
                else:
                    if meet_true:
                        #print(q[:10], a[:10], l, score)
                        meet_true = False

        evaluator = Evaluator(qa_pairs, scores)
        evaluator.calculate()

        print("map {:.4f}  mrr {:.4f} acc@1 {:.4f}".format(
            evaluator.MAP(), evaluator.MRR(), evaluator.ACC_at_1()))

    def input(self, batch_size, train_question_id_fn):
        print("create input from", train_question_id_fn)

        filename_queue = tf.train.string_input_producer([train_question_id_fn])
        reader = tf.TextLineReader(skip_header_lines=0)
        key, value = reader.read(filename_queue)
        decoded = tf.decode_csv(
                value,
                field_delim=' ',
                record_defaults=[[0]])
        # shuffle batches shape is [item_length, batch_size]
        shuffle_batches = tf.train.shuffle_batch(decoded,
                                  batch_size=batch_size,
                                  capacity=batch_size * 40,
                                  min_after_dequeue=batch_size)
        question_ids = tf.transpose(tf.stack(shuffle_batches))

        return question_ids

    def create_train_inputs(self, ids, questions, answers, train_json):
        """
        Args:
            ids: the quesiton id numpy array
            questions: questioin tokens np array
            answers: answer tokens np array
            train_json: the question, true answer, false answer map
        """

        ids = np.reshape(ids, -1)
        question_tokens = []
        true_tokens = []
        false_tokens = []

        for i in ids:
            true_answer_ids = train_json[str(i)]['answers']
            false_answer_ids = train_json[str(i)]['negatives']
            true_id = np.random.choice(true_answer_ids)
            false_id = np.random.choice(false_answer_ids)

            question_tokens.append(questions[int(i)])
            true_tokens.append(answers[int(true_id)])
            false_tokens.append(answers[int(false_id)])

            #print("q: {}, t: {}, f :{}".format(i, true_id, false_id))

        assert len(question_tokens) == len(true_tokens)
        assert len(question_tokens) == len(false_tokens)

        return question_tokens, true_tokens, false_tokens

    def create_train_eval_inputs_from_file(self, id_fn, questions, answers, train_json):
        """
        Args:
            id_fn: the eval question id numpy array
            questions: question tokens np array
            answers: answer token np array
            train_json: the question, true answer, false answer map
        """
        ids = []
        with open(id_fn, "r") as f:
            for line in f:
                line = line.strip()
                ids.append(int(line))

        question_tokens = []
        answer_tokens = []
        labels = []

        for i in ids:
            true_answer_ids = train_json[str(i)]['answers']
            false_answer_ids = train_json[str(i)]['negatives']

            for true_id in true_answer_ids:
                question_tokens.append(questions[int(i)])
                answer_tokens.append(answers[int(true_id)])
                labels.append(1)

            for false_id in false_answer_ids:
                question_tokens.append(questions[int(i)])
                answer_tokens.append(answers[int(false_id)])
                labels.append(0)

        assert len(question_tokens) == len(labels)
        assert len(answer_tokens) == len(labels)

        return np.array(question_tokens), np.array(answer_tokens), np.array(labels)


    def create_train_eval_inputs(self, ids, questions, answers, train_json):
        """
        Args:
            ids: the question id numpy array
            questions: question tokens np array
            answers: answer token np array
            train_json: the question, true answer, false answer map
        """
        ids = np.reshape(ids, -1)
        question_tokens = []
        answer_tokens = []
        labels = []

        for i in ids:
            true_answer_ids = train_json[str(i)]['answers']
            false_answer_ids = train_json[str(i)]['negatives']

            for true_id in true_answer_ids:
                question_tokens.append(questions[int(i)])
                answer_tokens.append(answers[int(true_id)])
                labels.append(1)

            for false_id in false_answer_ids:
                question_tokens.append(questions[int(i)])
                answer_tokens.append(answers[int(false_id)])
                labels.append(0)

        assert len(question_tokens) == len(labels)
        assert len(answer_tokens) == len(labels)

        return np.array(question_tokens), np.array(answer_tokens), np.array(labels)

    def train(self, batch_size, train_question_id_fn, train_question_token_fn,
            answer_token_fn, train_json_fn, validate_question_token_fn,
            validate_json_fn, log_dir,):
        """
        Args:
            train_question_id_fn: the train question id list file
            questions_token_fn: the question token fn
            answers_token_fn: the answer token fn
            train_json_fn: the train json input
        """
        insurance_loader = TokenLoader(self.max_sequence_length)
        question_tokens, answer_tokens, train_json = insurance_loader.load_tokens_for_queue(
                train_question_token_fn, answer_token_fn, train_json_fn)
        print("train queston num {}, answer num {}, train question json {}".format(len(question_tokens.keys()),
                    len(answer_tokens.keys()), len(train_json.keys())))

        #validate_questions, validate_answers, validate_labels = insurance_loader.load_tokens_for_validate(
        #                    validate_quesiton_token_fn, answer_token_fn, validate_json_fn)
        #print("validate question len", len(validate_questions))

        #validate_question_tokens, validate_answer_tokens, validate_json = insurance_loader.load_tokens_for_queue(
        #        validate_question_token_fn,
        #        answer_token_fn, validate_json_fn)
        train_question_all, train_answer_all, train_label_all = self.create_train_eval_inputs_from_file(
                FLAGS.train_question_id_eval_fn, question_tokens, answer_tokens,
                train_json)

        with self.graph.as_default():
            train_question_tensor = tf.placeholder(tf.int32, [batch_size,
                self.max_sequence_length])
            train_answer_tensor = tf.placeholder(tf.int32, [batch_size,
                self.max_sequence_length])
            train_false_answer_tensor = tf.placeholder(tf.int32, [batch_size,
                self.max_sequence_length])

            question_ids = self.input(batch_size, train_question_id_fn)

            loss = self.loss(train_question_tensor, train_answer_tensor,
                    train_false_answer_tensor)
            train_op = self.train_op(loss)

            valid_sim = self.validate_op()

            sv = tf.train.Supervisor(graph=self.graph, logdir=log_dir)

        with sv.managed_session(master='') as sess:
            for step in range(self.max_train_step):
                if sv.should_stop():
                    break
                question_ids_val = sess.run(question_ids)
                print(question_ids_val[:10])
                question_input, answer_input, false_answer_input = self.create_train_inputs(
                    question_ids_val, question_tokens, answer_tokens, train_json)
                #question_eval_tokens, answer_eval_tokens, eval_labels = self.create_train_eval_inputs(
                #    question_ids_val, question_tokens, answer_tokens, train_json)

                loss_val, _, true_sim_val, false_sim_val = sess.run([loss, train_op, self.true_sim,
                    self.false_sim],
                            {train_question_tensor: question_input,
                             train_answer_tensor: answer_input,
                             train_false_answer_tensor: false_answer_input})

                if (step + 1) % 10 == 0:
                    print("______loss {:.4f} step {}".format(loss_val, step + 1))
                    #print("true_sim ", true_sim_val)
                    #print("false_sim ", false_sim_val)
                    #print("subsract ", true_sim_val - false_sim_val)

                    #self.validate(sess, valid_sim, question_eval_tokens,
                    #        answer_eval_tokens, eval_labels)

                if  (step + 1) % 1000 == 0:
                    print("all validate")
                    #self.validate(sess, valid_sim, validate_questions[0:10000,:],
                    #        validate_answers[0:10000,:], validate_labels[0:10000])
                    self.validate(sess, valid_sim, train_question_all,
                            train_answer_all, train_label_all)

def main(_):
    w2v_model = w2v.load(FLAGS.word_vec_path)
    model = Model(w2v_model.vectors, FLAGS.max_sequence_length,
            FLAGS.filter_size, FLAGS.num_filter, FLAGS.learning_rate,
            FLAGS.margin, FLAGS.max_train_step)
    model.train(FLAGS.batch_size, FLAGS.train_question_id_fn,
            FLAGS.train_question_token_fn, FLAGS.answer_input, FLAGS.train_json_fn,
            FLAGS.validate_question_token_fn, FLAGS.validate_json_fn,
            FLAGS.log_dir)

if __name__ == "__main__":
    tf.app.run()
