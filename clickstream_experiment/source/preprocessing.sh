#!/bin/bash

for d in 20 100 250
do
  for e in 5 10 25
  do
    for l in 5 10 15 20
    do
      python3 preprocessing.py --w2v-dim d --w2v-epochs e --hmm-nodes n --w2v-min-len l
    done
  done
done