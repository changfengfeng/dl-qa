import gzip
import json
import numpy as np

"""
from qa_data import QALoader

class InsuranceData:
    def __init__(self, word_vec_path, max_sequence_length):
        self.word_vec_path = word_vec_path
        self.max_sequence_length = max_sequence_length
        self.qa_loader = QALoader(word_vec_path, max_sequence_length)

    def load_json(self, fn):
        with gzip.open(fn, 'rb') as f:
            data = json.loads(f.read())
        return data

    def load_text(self, fn):
        with gzip.open(fn, "rb") as f:
            return f.read()

    def generate_text(self, fn):
        texts = []
        valid_json = load_json(fn)
        for k, v in valid_json.items():
            texts.append(v['zh'])

        return texts

    def generate_train_data(self, train_fn, out):
        answer_json = load_json("data/pool/answers.json.gz")

        train_json = load_json(train_fn)
        for k, v in train_json.items():
            question = v['zh']
            answers = v['answers']
            false_answers = v['negatives']

            question_id = self.qa_loader.sentence_to_ids(question).astype(str)

            for a in answers:
                answer_id = self.qa_loader.sentence_to_ids(answer_json[a]['zh']).astype(str)
                print(k, " ".join(question_id), " ".join(answer_id), 1, file=out)

            for a in false_answers:
                answer_id = self.qa_loader.sentence_to_ids(answer_json[a]['zh']).astype(str)
                print(k, " ".join(question_id), " ".join(answer_id), 0, file=out)

    def generate_answer_tokens(self, answer_fn, out):
        answer_json = self.load_json(answer_fn)
        for k, v in answer_json.items():
            ids = self.qa_loader.sentence_to_ids(v['zh'])
            print(k, " ".join(ids.astype(str)), file=out)

    def generate_question_tokens(self, question_fn, out):
        question_json = self.load_json(question_fn)
        for k, v in question_json.items():
            ids = self.qa_loader.sentence_to_ids(v['zh'])
            print(k, " ".join(ids.astype(str)), file=out)

    def generate_train_question_ids(self, train_json_fn, out):
        train_json = self.load_json(train_json_fn)
        for k, v in train_json.items():
            print(k, file=out)

"""
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
                assert len(items) == 2 * self.max_sequence_length + 2
                questions.append(items[1:self.max_sequence_length + 1])
                answers.append(items[self.max_sequence_length + 1:-1])
                label = int(items[-1])
                labels.append(label)

        return np.array(questions, "int32"), np.array(answers, "int32"), np.array(labels, "int32")

    def load_tokens_for_queue(self, question_token_fn, answer_token_fn,
            train_json_fn):
        print("load token for queue", question_token_fn, answer_token_fn,
                train_json_fn)
        with open(question_token_fn, "r") as f:
            questions = {}
            for line in f:
                line = line.strip()
                items = line.split(" ")
                questions[int(items[0])] = np.array(items[1:], "int32")

        with open(answer_token_fn, "r") as af:
            answers = {}
            for line in af:
                line = line.strip()
                items = line.split(" ")
                answers[int(items[0])] = np.array(items[1:], "int32")

        with gzip.open(train_json_fn, 'rb') as gf:
            data = json.loads(gf.read())

        return questions, answers, data

    def load_tokens_for_validate(self, quesiton_token_fn, answer_token_fn,
            validate_json_fn):
        print("load valid", quesiton_token_fn, answer_token_fn,
                validate_json_fn)
        questions, answers, validate_json = self.load_tokens_for_queue(
                quesiton_token_fn, answer_token_fn, validate_json_fn)

        question_tokens = []
        answer_tokens = []
        labels = []

        for k, v in validate_json.items():
            q_id = int(k)
            true_answers = v['answers']
            false_answers = v['negatives']

            for a in true_answers:
                question_tokens.append(questions[q_id])
                answer_tokens.append(answers[int(a)])
                labels.append(1)

            for a in false_answers:
                question_tokens.append(questions[q_id])
                answer_tokens.append(answers[int(a)])
                labels.append(0)

        assert len(question_tokens) == len(answer_tokens)
        assert len(question_tokens) == len(labels)

        return np.array(question_tokens), np.array(answer_tokens), np.array(labels)

    def load_valid_json(self, valid_json_fn):
        with gzip.open(valid_json_fn, "rb") as f:
            return json.loads(f.read())

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
                assert len(items) == 2 * self.max_sequence_length + 2
                question = items[1:self.max_sequence_length + 1]
                answer = items[self.max_sequence_length+1:-1]
                label = int(items[-1])

                question_id = items[0]
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
