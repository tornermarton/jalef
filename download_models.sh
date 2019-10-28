#!/usr/bin/env bash

mkdir models/
mkdir models/bert/
mkdir models/word2vec/

wget -O "models/bert_model.tar.gz" "https://tfhub.dev/google/bert_cased_L-24_H-1024_A-16/1?tf-hub-format=compressed"
tar --directory=models/bert/ -xzf models/bert_model.tar.gz
rm -rf models/bert_model.tar.gz

wget --directory-prefix=models/word2vec/ -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"