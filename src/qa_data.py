# -*- coding: utf-8 -*-
import sys
sys.path.append("../dl-segmentor/src")

import ner
import tensorflow as tf
import word2vec as w2v
import numpy as np

FLAGS = tf.flags.FLAGS

class QALoader:
    def __init__(self, word_vec_path, max_sequence_length):
        self.word_vec_path = word_vec_path
        self.max_sequence_length = max_sequence_length
        # 80 cha per sub sentence
        self.segmentor = ner.Segmentor(FLAGS.user_dict_path, FLAGS.kcws_char_vocab_path,
            FLAGS.segment_model_path, "segment", 80)

        self.w2v_model = w2v.load(word_vec_path)
        self.word_lexicon = {w:i for i, w in enumerate(self.w2v_model.vocab)}

    def load_data(self, filename, column):
        question = ""
        questions, answers, labels = [], [], []
        with open(filename, mode="r", encoding="utf-8") as rf:
            for line in rf.readlines():
                arr = line.split("\t")
                if len(arr) == column:
                    questions.append(arr[0])
                    answers.append(arr[1])
                    if column == 3:
                        labels.append(int(arr[2]))

        assert len(questions) == len(answers)

        print("load {} from {}".format(len(questions), filename))
        return questions, answers, labels

    def sentence_to_ids(self, sentence):
        tokens = self.segmentor.segment(sentence)
        ids = np.zeros(self.max_sequence_length, dtype="int32")
        max_token_num = len(tokens)
        if max_token_num > self.max_sequence_length:
            max_token_num = self.max_sequence_length
        for i in range(max_token_num):
            token = tokens[i]
            if token in self.word_lexicon:
                ids[i] = self.word_lexicon[token]
            else:
                ids[i] = self.word_lexicon['<UNK>']
        return ids

class TokenLoader:
    def __init__(self, max_sequence_length):
        self.max_sequence_length = max_sequence_length

    def load_tokens(self, filename):
        questions = []
        answers = []
        labels = []
        with open(filename, "r") as f:
            for line in f:
                items = line.split(" ")
                assert len(items) == 2 * self.max_sequence_length + 1
                questions.append(items[:self.max_sequence_length])
                answers.append(items[self.max_sequence_length:2 *
                    self.max_sequence_length])
                label = int(items[-1])
                labels.append(label)

        return np.array(questions, "int32"), np.array(answers, "int32"), np.array(labels, "int32")

    def load_true_false(self, filename):
        """ load question, answer, wrong triples
        """
        questions = []
        true_answers = []
        wrong_answers = []
        last_question = ""
        true_answer = None
        wrong_count = 0

        with open(filename, "r") as f:
            for line in f:

                items = line.split(" ")
                assert len(items) == 2 * self.max_sequence_length + 1
                question = items[:self.max_sequence_length]
                answer = items[self.max_sequence_length:-1]
                label = int(items[-1])

                question_id = "".join(question)
                if question_id != last_question:
                    if true_answer != None:
                        true_answers.extend([true_answer] * wrong_count)
                    elif last_question != "":
                        questions = questions[:-wrong_count]
                        wrong_answers = wrong_answers[:-wrong_count]
                    true_answer = None
                    last_question = question_id
                    wrong_count = 0

                if label == 0:
                    wrong_count += 1
                    wrong_answers.append(answer)
                    questions.append(question)
                else:
                    true_answer = answer

        if true_answer != None:
            true_answers.extend([true_answer] * wrong_count)

        assert len(questions) == len(true_answers)
        assert len(questions) == len(wrong_answers)

        return np.array(questions, "int32"), np.array(true_answers, "int32"), np.array(wrong_answers, "int32")
