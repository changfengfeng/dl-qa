# -*- coding: utf-8 -*-
import sys
sys.path.append("../dl-segmentor/src")

import os
import tensorflow as tf
from qa_data import QALoader

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("word_vec_path", "", "")
qa_loader = QALoader(FLAGS.word_vec_path, 80)

long_line = 0
total_line = 0

train_fn = "data/qa/training.data"
train_id_fn = "data/qa/train_id.data"

test_fn = "data/qa/testing.data"
test_id_fn = "data/qa/test_id.data"

validate_fn = "data/qa/develop.data"
validate_id_fn = "data/qa/validate_id.data"

questions, answers, labels = qa_loader.load_data(train_fn, 3)
out = open(train_id_fn, "w")
for q, a, l in zip(questions, answers, labels):
    q_ids = qa_loader.sentence_to_ids(q)
    a_ids = qa_loader.sentence_to_ids(a)
    out.write(" ".join(q_ids.astype(str)))
    out.write(" ")
    out.write(" ".join(a_ids.astype(str)))
    out.write(" %d\n" % l)
    assert len(q_ids) == 80
    assert len(a_ids) == 80

questions, answers, labels = qa_loader.load_data(test_fn, 2)
out = open(test_id_fn, "w")
for q, a in zip(questions, answers):
    q_ids = qa_loader.sentence_to_ids(q)
    a_ids = qa_loader.sentence_to_ids(a)
    out.write(" ".join(q_ids.astype(str)))
    out.write(" ")
    out.write(" ".join(a_ids.astype(str)))
    out.write("\n")
    assert len(q_ids) == 80
    assert len(a_ids) == 80

questions, answers, labels = qa_loader.load_data(validate_fn, 3)
out = open(validate_id_fn, "w")
for q, a, l in zip(questions, answers, labels):
    q_ids = qa_loader.sentence_to_ids(q)
    a_ids = qa_loader.sentence_to_ids(a)
    out.write(" ".join(q_ids.astype(str)))
    out.write(" ")
    out.write(" ".join(a_ids.astype(str)))
    out.write(" %d\n" % l)
    assert len(q_ids) == 80
    assert len(a_ids) == 80
