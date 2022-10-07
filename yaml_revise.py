from ruamel import yaml
import argparse
import os

parser = argparse.ArgumentParser(description="template revise.")
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
        with open(os.path.join(template_yaml_path, name), 'r', encoding='utf-8') as f:
            raw_cfg = f.read()
            cfg = yaml.load(raw_cfg, Loader=yaml.Loader)  # type:dict
        for gum in list(range(2, 33, 2)):
            cfg["agent_para"]["agent_config"]["gum"] = gum
            cfg["agent_para"]["agent_config"]["complete_step"] = 400000
            cfg['benchmark']['id'] = "ppo_batch{}".format(gum)
            with open(os.path.join(train_yaml_path, "{}.template".format(cfg['benchmark']['id'])), 'w',
                      encoding='utf-8') as f:
                yaml.dump(cfg, f, Dumper=yaml.RoundTripDumper)


if __name__ == '__main__':
    yaml_revise(args.template, args.target)
