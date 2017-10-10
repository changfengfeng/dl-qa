# -*- coding: utf-8 -*-
import sys
sys.path.append("../dl-segmentor/src")

import os
from ner import Segmentor
import tensorflow as tf

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("input_dir", "", "")
tf.flags.DEFINE_string("output", "", "")

segmentor = Segmentor(FLAGS.user_dict_path, FLAGS.kcws_char_vocab_path,
            FLAGS.segment_model_path, "segment", 80)

long_line = 0
total_line = 0
def process_line(line, out):
    global long_line
    global total_line
    if len(line) > 80:
        line = line[:80]
        long_line += 1
    total_line += 1
    tokens = segmentor.segment(line)
    out.write(" ".join(tokens))
    out.write("\n")

def main(argc, argv):
  global long_line
  global total_line
  rootDir = FLAGS.input_dir
  out = open(FLAGS.output, "w")
  for dirName, subdirList, fileList in os.walk(rootDir):
    curDir = dirName
    for file in fileList:
      curFile = os.path.join(curDir, file)
      print("processing:%s" % (curFile))
      fp = open(curFile, mode="r", encoding="utf-8")
      last_question = ""
      for line in fp.readlines():
        lines = line.strip().split("\t")
        if len(lines) > 1:
            question = lines[0].strip()
            answer = lines[1].strip()
            if question != last_question:
                process_line(question, out)
                last_question = question
            process_line(lines[1], out)
        else:
            print(line)
      fp.close()
  out.close()

  print(total_line, long_line)

if __name__ == '__main__':
  main(len(sys.argv), sys.argv)
