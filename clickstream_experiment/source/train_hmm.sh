#!/bin/bash

for d in 20 100 250
do
  for e in 5 10 25
  do
    for l in 5 10 15 20
    do
      for n in 10 50 100 1000 10000
      do
        python3 train_hmm.py --w2v-dim $d --w2v-epochs $e --w2v-min-len $l --hmm-nodes $n -hmm-min-len $l
      done
    done
  done
done