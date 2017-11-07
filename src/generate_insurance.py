import tensorflow as tf
from insurance_data import InsuranceData

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("word_vec_path", "data/word_vec.txt", "")
tf.flags.DEFINE_integer("max_sequence_length", 200, "")

insurance_data = InsuranceData(FLAGS.word_vec_path,
        FLAGS.max_sequence_length)


with open("data/insurance_train_question", "w") as out:
    insurance_data.generate_question_tokens("data/pool/train.json.gz", out)

with open("data/insurance_valid_question", "w") as out:
    insurance_data.generate_question_tokens("data/pool/valid.json.gz", out)

with open("data/insurance_test_question", "w") as out:
    insurance_data.generate_question_tokens("data/pool/test.json.gz", out)
"""
with open("data/insurance_answer", "w") as out:
    insurance_data.generate_answer_tokens("data/pool/answers.json.gz", out)

with open("data/insurance_train_question_id", "w") as out:
    insurance_data.generate_train_question_ids("data/pool/train.json.gz", out)
"""

