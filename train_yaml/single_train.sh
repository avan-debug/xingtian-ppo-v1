#!/bin/bash
# conda activate xingtian
rm -f run.log
xt_main -f $1 &>run.log
./kill.sh
wait
echo "complete."
