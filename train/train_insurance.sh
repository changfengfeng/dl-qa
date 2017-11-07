#! /bin/sh

source ~/.bash_profile

echo "_____prepare"
#python src/process_insurance.py \
#    --segment_model_path=../dl-segmentor/model/segment_model_kcws.pbtxt \
#    --kcws_char_vocab_path=../dl-segmentor/model/kcws_char_pepole_vec.txt \
#    --user_dict_path=../dl-segmentor/model/user_dict.txt \
#    --input_dir=data/pool \
#    --output=data/insurance.txt

#wc -l data/pepole.txt
#wc -l data/insurance.txt
#cat data/insurance.txt>> data/pepole.txt
#wc -l data/pepole.txt

echo "_____word vocab"
#src/word2vec/word2vec \
#    -train data/pepole.txt \
#    -save-vocab data/pepole_vocab.txt \
#    -min-count 5

echo "_____unk"
#python src/replace_unk.py data/pepole_vocab.txt \
#    data/pepole.txt \
#    data/pepole_unk.txt

echo "_____word2vec"
#src/word2vec/word2vec -train data/pepole_unk.txt -output data/word_vec.txt -size 150 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0  -cbow 0 -iter 3 -min-count 5 -hs 1

echo "_____generate training data"
python src/generate_insurance.py  \
    --segment_model_path=../dl-segmentor/model/segment_model_kcws.pbtxt \
    --kcws_char_vocab_path=../dl-segmentor/model/kcws_char_pepole_vec.txt \
    --user_dict_path=../dl-segmentor/model/user_dict.txt \
    --word_vec_path=data/word_vec.txt \
    --max_sequence_length=200
