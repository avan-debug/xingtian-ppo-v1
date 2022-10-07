#!/bin/bash
list=$(ls $1)
for l in $list
do
  echo "$l"
  ./single_train.sh $1/$l
  echo "complete $l."
  wait

done
