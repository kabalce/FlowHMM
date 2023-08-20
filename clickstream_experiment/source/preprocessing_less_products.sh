#!/bin/bash

for d in 20 100 250
do
  for e in 5 10 25
  do
    for l in 5 10 15 20
    do
      python3 preprocessing_less_products.py --w2v-dim $d --w2v-epochs $e --w2v-min-len $l
    done
  done
done