#!/bin/bash
list=$(ls $1|grep -v "monitor")
for l in $list
do
  echo "$l"
  xt_main -f $1/$l &> run.log
  ./kill.sh
  wait
  echo "complete $l."
  wait

done
