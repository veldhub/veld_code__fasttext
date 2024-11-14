#!/bin/bash

# launching the python script via this bash script is a work-around since otherwise the fasttext
# library wouldn't print properly to stdout when executed in a docker container, likely due to some
# internal unchangeable buffering. Unbuffering python didn't work at all, so the entire stdout of
# the python call is piped into a tmp file which is looped over and returned to the screen, until
# the training process has finished.

echo "in_train_data_file: ${in_train_data_file}"
echo "out_model_file: ${out_model_file}"
echo "vector_size: ${vector_size}"
echo "epochs: ${epochs}"
echo "window_size: ${window_size}"

python /veld/code/train.py &> /tmp/log &

pid=$!

sleep 5

cat /tmp/log

while ps -p $pid > /dev/null
do
  sleep 3
  tail -n 2 /tmp/log
done

