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
        if len(sentence) > self.max_sequence_length:
            sentence = sentence[:self.max_sequence_length]
        tokens = self.segmentor.segment(sentence)
        ids = np.zeros(self.max_sequence_length, dtype="int32")
        for i in range(len(tokens)):
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
                print(len(items))
                assert len(items) == 2 * self.max_sequence_length + 1
                questions.append(items[:self.max_sequence_length])
                answers.append(items[self.max_sequence_length:2 *
                    self.max_sequence_length])
                labels.append(items[-1])

        return np.array(questions, "int32"), np.array(answers, "int32"), np.array(labels, "int32")

if __name__ == "__main__":
    loader = QALoader("../dl-segmentor/data/ner_pepole_vec.txt", 50)
    questions, answers, labels = loader.load_data("data/training.data")
    for q in questions:
        ids = loader.sentence_to_ids(q)
        print(ids)

