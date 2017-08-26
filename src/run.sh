#!/bin/bash

python3 generate_train.py ../data/train.csv ../data/test.csv ../data/full/train_sub.csv
python3 preprocess.py ../data/full/train_sub.csv ../data/test.csv ../data/full/train_proc.csv ../data/full/test_proc.csv n
python3 gen_data.py ../data/full/train_sub_proc.csv ../data/full/test_proc.csv ../data/full/train_sub_proc_count.csv ../data/full/test_proc_count.csv t f all
python3 converter.py ../data/full/train_sub_proc_count.csv ../data/full/test_proc_count.csv ../data/full/train.ffm ../data/full/test.ffm t f all n n x x

cd ../models/full

../../libffm/ffm-train -k 4 -s 12 -t 13 ../../data/full/train.ffm train.model

cd ../predictions/full

../../libffm/ffm-predict ../../data/full/test.ffm ../../models/full/train.model test_ffm.txt

cd ../../src

python3 save_preds.py ../data/test.csv ../predictions/full/test_ffm.txt ../submissions/ffm_new.csv
