import re
from ruamel import yaml
import argparse
import os
import numpy as np
from collections import defaultdict

parser = argparse.ArgumentParser(description="template generator.")
parser.add_argument("-r", "--template", type=str, default="./template")
parser.add_argument("-o", "--target", type=str, default="./train_yaml")
parser.add_argument("-t", "--type", type=str, default="multi")
parser.add_argument("-n", "--step", type=int, default=1000000)
# parser.add_argument("-g", "--gum", type=int, default=2)
args = parser.parse_args()


def yaml_revise(template_yaml_path, train_yaml_path):
    names = os.listdir(template_yaml_path)
    if not os.path.exists(train_yaml_path):
        os.makedirs(train_yaml_path)
    for name in names:
        PREFIX = "PIPELINE" if re.search("BENCH", name, re.I) is None else "BENCH"

        if PREFIX == "BENCH":
            continue
        with open(os.path.join(template_yaml_path, name), 'r', encoding='utf-8') as f:
            raw_cfg = f.read()
            cfg = yaml.load(raw_cfg, Loader=yaml.Loader)

        for env_num in list(range(10, 81, 10)):
            if PREFIX == "BENCH":
                cfg["env_num"] = env_num
            else:
                cfg["env_num"] = 3
                cfg["env_para"]["env_info"]["size"] = env_num
                cfg["env_para"]["env_info"]["wait_nums"] = env_num - 2
                cfg["alg_para"]["alg_config"]["prepare_times_per_train"] = env_num * 3

            cfg["agent_para"]["agent_config"]["complete_step"] = 10000000
            cfg['benchmark']['id'] = PREFIX + "_{}".format(env_num)

            with open(os.path.join(train_yaml_path, "{}.yaml".format(cfg['benchmark']['id'])), 'w',
                      encoding='utf-8') as f:
                yaml.dump(cfg, f, Dumper=yaml.RoundTripDumper)


if __name__ == '__main__':
    # yaml_revise(args.template, args.target)
    yaml_revise("./template", "./yaml")
