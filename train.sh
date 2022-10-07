#/bin/bash
a=(2 3 4 5)
for i in ${a[*]};
do
    file=breakout_ppo$i.yaml
    echo $file
    ./single_train.sh ./diff_group_perf_test/$file
done
