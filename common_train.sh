#!/bin/bash
list=$(ls $1|grep -v "monitor")
for l in $list
do
  echo "$l"
  ./single_train.sh $1/$l
  echo "complete $l."
  wait

done
