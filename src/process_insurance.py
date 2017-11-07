# -*- coding: utf-8 -*-
import sys
sys.path.append("../dl-segmentor/src")

import os
from ner import Segmentor
import tensorflow as tf
import insurance_data

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("input_dir", "", "")
tf.flags.DEFINE_string("output", "", "")

segmentor = Segmentor(FLAGS.user_dict_path, FLAGS.kcws_char_vocab_path,
            FLAGS.segment_model_path, "segment", 80)

total_line = 0
def process_line(line, out):
    global total_line
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
      if curFile.endswith("json.gz"):
        print("processing:%s" % (curFile))
        lines = insurance_data.generate_text(curFile)
        for line in lines:
            line = line.strip()
            process_line(line, out)
      print(total_line, segmentor.long_line)

  out.close()

if __name__ == '__main__':
  main(len(sys.argv), sys.argv)
