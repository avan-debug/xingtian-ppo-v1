for name in $(ls ./train_yaml)
do
    python xt/main.py -f ./train_yaml/$name
    wait
    ./kill.sh
done

