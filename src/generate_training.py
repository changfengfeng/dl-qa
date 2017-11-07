# -*- coding: utf-8 -*-
import sys
sys.path.append("../dl-segmentor/src")

import os
import tensorflow as tf
from qa_data import QALoader
import json
import gzip

max_word_num = 150
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("word_vec_path", "", "")
qa_loader = QALoader(FLAGS.word_vec_path, max_word_num)

long_line = 0
total_line = 0

answer_token_fn = "data/qa/answer_token"

train_fn = "data/qa/training.data"
train_id_fn = "data/qa/train_id.data"
train_json_fn = "data/qa/train.json.gz"
train_question_id_fn = "data/qa/train_question_id"
train_question_token_fn = "data/qa/train_question_token"

validate_fn = "data/qa/develop.data"
validate_id_fn = "data/qa/validate_id.data"
validate_json_fn = "data/qa/validate.json.gz"
validate_token_fn = "data/qa/validate_question_token"

answer_map = {}
answer_id = 0

questions, answers, labels = qa_loader.load_data(train_fn, 3)
"""
out = open(train_id_fn, "w")
for q, a, l in zip(questions, answers, labels):
    q_ids = qa_loader.sentence_to_ids(q)
    a_ids = qa_loader.sentence_to_ids(a)
    out.write(" ".join(q_ids.astype(str)))
    out.write(" ")
    out.write(" ".join(a_ids.astype(str)))
    out.write(" %d\n" % l)
    assert len(q_ids) == max_word_num
    assert len(a_ids) == max_word_num
"""
qid_out = open(train_question_id_fn, "w")
qtoken_out = open(train_question_token_fn, "w")
qa_pairs = {}
for q, a, l in zip(questions, answers, labels):
    q = q.strip()
    if a in answer_map:
        a_id = answer_map[a]
    else:
        a_id = answer_id
        answer_id += 1
        answer_map[a] = a_id
    if q in qa_pairs:
        if l == 1:
            qa_pairs[q]['answers'].append(a_id)
        else:
            qa_pairs[q]['negatives'].append(a_id)
    else:
        if l == 1:
            qa_pairs[q] = {'question':q, 'answers':[a_id], 'negatives':[]}
        else:
            qa_pairs[q] = {'question':q, 'answers':[], 'negatives':[a_id]}

q_id = 0
out_qa_pairs = {}
for k, v in qa_pairs.items():
    out_qa_pairs[q_id] = v
    qtokens = qa_loader.sentence_to_ids(k)
    qtokens = " ".join(qtokens.astype(str))
    qid_out.write("%d\n" % q_id)
    qtoken_out.write("%d %s\n" % (q_id, qtokens))
    q_id += 1
qid_out.close()

with gzip.open(train_json_fn, "w") as gf:
    gf.write(json.dumps(out_qa_pairs, ensure_ascii=False).encode('utf-8'))

# validate data
questions, answers, labels = qa_loader.load_data(validate_fn, 3)
"""
out = open(validate_id_fn, "w")
for q, a, l in zip(questions, answers, labels):
    q_ids = qa_loader.sentence_to_ids(q)
    a_ids = qa_loader.sentence_to_ids(a)
    out.write(" ".join(q_ids.astype(str)))
    out.write(" ")
    out.write(" ".join(a_ids.astype(str)))
    out.write(" %d\n" % l)
    assert len(q_ids) == max_word_num
    assert len(a_ids) == max_word_num
"""
qtoken_out = open(validate_token_fn, "w")
qa_pairs = {}
for q, a, l in zip(questions, answers, labels):
    q = q.strip()
    if a in answer_map:
        a_id = answer_map[a]
    else:
        a_id = answer_id
        answer_id += 1
        answer_map[a] = a_id
    if q in qa_pairs:
        if l == 1:
            qa_pairs[q]['answers'].append(a_id)
        else:
            qa_pairs[q]['negatives'].append(a_id)
    else:
        if l == 1:
            qa_pairs[q] = {'question':q, 'answers':[a_id], 'negatives':[]}
        else:
            qa_pairs[q] = {'question':q, 'answers':[], 'negatives':[a_id]}

q_id = 0
out_qa_pairs = {}
for k, v in qa_pairs.items():
    out_qa_pairs[q_id] = v
    qtokens = qa_loader.sentence_to_ids(k)
    qtokens = " ".join(qtokens.astype(str))
    qtoken_out.write("%d %s\n" % (q_id, qtokens))
    q_id += 1
qid_out.close()

with gzip.open(validate_json_fn, "w") as gf:
    gf.write(json.dumps(out_qa_pairs, ensure_ascii=False).encode('utf-8'))

"""
with open(answer_token_fn, "w") as gf:
    for k, v in answer_map.items():
        answer_tokens = qa_loader.sentence_to_ids(k)
        answer_tokens = " ".join(answer_tokens.astype(str))
        gf.write("%d %s\n" % (v, answer_tokens))
"""
