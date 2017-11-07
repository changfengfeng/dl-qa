#!/usr/bin/python
# coding=utf-8
""" The implementation of calculating MRR/MAP/ACC@1
    Usage: 'python evaluation.py QApairFile scoreFile outputFile'
"""
import numpy as np
import gzip
import json

class Evaluator(object):

    def __init__(self, qaPairs, scores):
        self.qIndex2aIndex2aScore = {}
        self.qIndex2aIndex2aLabel = {}
        self.ACC_at1List = []
        self.APlist = []
        self.RRlist = []
        self.loadData(qaPairs, scores)

    def loadData(self, qaPairs, scores):
        assert len(qaPairs) == len(scores)
        qIndex = 0
        aIndex = 0
        label = 0
        score = 0.0
        lastQuestion = ''
        question = ''

        for idx in range(len(qaPairs)):
            qaLineArr = qaPairs[idx]
            assert len(qaLineArr) == 3
            question = "".join(qaLineArr[0])
            label = int(qaLineArr[2])
            #print(scores[idx])
            score = float(scores[idx])
            if question != lastQuestion:

                if idx != 0:
                    qIndex += 1
                aIndex = 0
                lastQuestion = question
            if not qIndex in self.qIndex2aIndex2aScore:
                self.qIndex2aIndex2aScore[qIndex] = {}
                self.qIndex2aIndex2aLabel[qIndex] = {}
            self.qIndex2aIndex2aLabel[qIndex][aIndex] = label
            self.qIndex2aIndex2aScore[qIndex][aIndex] = score

            #print("q_index {}, a_index {} new question {}, last question {}, ".format(
            #        question, lastQuestion, qIndex, aIndex))

            aIndex += 1

    def calculate(self):
        for qIndex, index2scoreList in self.qIndex2aIndex2aScore.items():
            index2label = self.qIndex2aIndex2aLabel[qIndex]

            rankIndex = 0
            rightNum = 0
            curPList = []
            rankedList = sorted(index2scoreList.items(), key=lambda b: b[1], reverse=True)
            self.ACC_at1List.append(0)
            index = 0
            for info in rankedList:
                aIndex = info[0]
                label = index2label[aIndex]
                rankIndex += 1
                if label == 1:
                    rightNum += 1
                    if rankIndex == 1:
                        self.ACC_at1List[-1] = 1
                    p = float(rightNum) / rankIndex
                    curPList.append(p)
                if index < 10:
                    #print(qIndex, aIndex, label, info[1])
                    index += 1
            if len(curPList) > 0 and len(curPList) != len(rankedList):
                self.RRlist.append(curPList[0])
                self.APlist.append(float(sum(curPList)) / len(curPList))
            else:
                self.ACC_at1List.pop()

    def MRR(self):
        return float(sum(self.RRlist)) / len(self.RRlist)

    def MAP(self):
        return float(sum(self.APlist)) / len(self.APlist)

    def ACC_at_1(self):
        return float(sum(self.ACC_at1List)) / len(self.ACC_at1List)

if __name__ == "__main__":
    """
    qa_pairs = []
    qa_pairs.append([np.array([1,2]).astype(str), np.array([5,6]).astype(str), 0])
    qa_pairs.append([np.array([1,2]).astype(str), np.array([3,4]).astype(str), 1])
    qa_pairs.append([np.array([1,2]).astype(str), np.array([7,8]).astype(str), 0])

    qa_pairs.append([np.array([2,1]).astype(str), np.array([5,6]).astype(str), 0])
    qa_pairs.append([np.array([2,1]).astype(str), np.array([7,8]).astype(str), 0])
    qa_pairs.append([np.array([2,1]).astype(str), np.array([3,4]).astype(str), 1])

    scores = [0.6, 0.8, 0.7, 0.8, 0.7, 0.9]
    """
    qa_pairs = []
    scores = []
    with gzip.open("data/pool/valid.json.gz", "rb") as f:
        data = json.loads(f.read())
        for q, ans in data.items():
            for r in ans['answers']:
                qa_pairs.append([np.array([q]).astype(str),
                    np.array([r]).astype(str), 1])
                scores.append(np.random.uniform())
            for r in ans['negatives']:
                qa_pairs.append([np.array([q]).astype(str),
                    np.array([r]).astype(str), 0])
                scores.append(np.random.uniform())

    calc = Evaluator(qa_pairs, scores)
    calc.calculate()

    print(calc.MAP(), calc.MRR(), calc.ACC_at_1())



