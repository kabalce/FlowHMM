#!/bin/bash

for d in 10, 50, 100, 250
do
  for e in 1, 5, 10
  do
    for n in 100, 1000, 5000, 10000
    do
      for l in 5, 10, 15, 20
      do
        python3 preprocessing.py --w2v-dim d --w2v-epochs e --hmm-nodes n --w2v-min-len l
      done
    done
  done
done