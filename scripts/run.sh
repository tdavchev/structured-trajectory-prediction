#!/bin/bash

num_mix='1 2 5 10 15 20'
rnn_size='256 512 768'

for mix in $num_mix
do
  for size in $rnn_size
  do
    python train_rnn.py $mix $size
    sleep 1.5
  done
done
