#!/bin/bash

for d in 100 250
do
  for e in 10 25
  do
    for l in 10 15
    do
      for n in 100 1000 10000
      do
        python3 train_hmm.py --w2v-dim $d --w2v-epochs $e --w2v-min-len $l --hmm-nodes $n --hmm-min-len $l --n-components 20
      done
    done
  done
done