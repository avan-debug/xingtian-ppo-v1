xt_main -f /home/xys/xingtian-ppo-v1/train_yaml/breakout_ppo.yaml
xt_main -f /home/xys/xingtian-ppo-v1/train_yaml/breakout_ppo2.yaml
pip install -e /home/xys/xingtian-test/xingtian-master3/xingtian-master
xt_main -f /home/xys/xingtian-test/xingtian-master3/xingtian-master/train_yaml/breakout_ppo_not_block.yaml
xt_main -f /home/xys/xingtian-test/xingtian-master3/xingtian-master/train_yaml/breakout_ppo.yaml


