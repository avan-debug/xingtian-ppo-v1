import datetime
import os
from time import time
from venv import create
import yaml

def create_yaml(file_name, update_dic, abs_path, str_time):
    with open(file_name) as f:
        doc = yaml.safe_load(f)
    
    doc["env_para"]["env_info"]["size"] = update_dic["size"]
    doc["env_para"]["env_info"]["wait_nums"] = update_dic["wait_nums"]

    doc["env_num"] = update_dic["env_num"]

    str_ = doc["model_para"]["actor"]["model_name"]
    for key, val in update_dic.items():
        str_ += "_" + str(key) + "_" + str(val)
    
    doc["benchmark"].update({"id": str_})
    doc["benchmark"].update({"archive_root": doc["benchmark"]["archive_root"] + str_time})

    with open(abs_path + doc["benchmark"]["id"] + ".yaml", 'w') as f:
        yaml.safe_dump(doc, f, default_flow_style=False)

def create_yaml_LR(file_name, update_dic, abs_path):
    with open(file_name) as f:
        doc = yaml.safe_load(f)

    str_ = doc["model_para"]["actor"]["model_name"]
    for key, val in update_dic.items():
        str_ += "_" + str(key) + "_" + str(val)
    doc["benchmark"].update({"id": str_})

    doc["model_para"]["actor"]["model_config"].update(update_dic)
    with open(abs_path + doc["benchmark"]["id"] + ".yaml", 'w') as f:
        yaml.safe_dump(doc, f, default_flow_style=False)

if __name__ == "__main__":
    abs_path = "/home/xys/xingtian-ppo-v1/train_yaml/train_ppo_dif_inf/"
    os.mkdir(abs_path)
    # str_time = str(datetime.datetime.now().strftime('%F_%T'))
    # file_name_ppo = "/home/xys/xingtian-ppo-v1/train_yaml/breakout_ppo.yaml"
    file_name_ppolite = "/home/xys/xingtian-ppo-v1/train_yaml/breakout_ppo2.yaml"

    update_arrs = []
    # update_dic1 = {"size": 5, "wait_nums": 3, "env_num": 5}
    # update_dic2 = {"size": 5, "wait_nums": 4, "env_num": 5}
    # update_dic3 = {"size": 6, "wait_nums": 3, "env_num": 4}
    # update_dic4 = {"size": 6, "wait_nums": 4, "env_num": 4}
    # update_dic5 = {"size": 6, "wait_nums": 5, "env_num": 4}
    # update_dic6 = {"size": 7, "wait_nums": 4, "env_num": 3}
    # update_dic7 = {"size": 7, "wait_nums": 5, "env_num": 3}
    # update_dic8 = {"size": 7, "wait_nums": 6, "env_num": 3}

    

    # update_arrs.append(update_dic1)
    # update_arrs.append(update_dic2)
    # update_arrs.append(update_dic3)
    # update_arrs.append(update_dic4)
    # update_arrs.append(update_dic5)
    # update_arrs.append(update_dic6)
    # update_arrs.append(update_dic7)
    # update_arrs.append(update_dic8)

    # for dic in update_arrs:
    #     create_yaml(file_name_ppo, dic, abs_path, str_time)
    # for dic in update_arrs:
    #     create_yaml(file_name_ppolite, dic, abs_path, str_time)
    update_dic1 = {"LR": 0.00035}
    update_dic2 = {"LR": 0.00045}
    update_dic3 = {"LR": 0.00055}
    # update_dic2 = {"LR": 0.00055}
    update_dic4 = {"LR": 0.00065}
    # update_dic3 = {"LR": 0.00075}
    # update_dic1 = {"LR": 0.0001}
    # update_dic2 = {"LR": 0.00025}
    # update_dic3 = {"LR": 0.0005}
    # update_dic4 = {"LR": 0.00075}
    # update_dic8 = {"LR": 0.001}
    # update_dic5 = {"LR": 0.00125}
    # update_dic6 = {"LR": 0.0015}
    # update_dic7 = {"LR": 0.00175}

    update_arrs.append(update_dic1)
    update_arrs.append(update_dic2)
    update_arrs.append(update_dic3)
    update_arrs.append(update_dic4)

    # update_arrs.append(update_dic3)
    # update_arrs.append(update_dic4)
    # update_arrs.append(update_dic5)
    # update_arrs.append(update_dic6)
    # update_arrs.append(update_dic7)
    # update_arrs.append(update_dic8)

    for dic in update_arrs:
        create_yaml_LR(file_name_ppolite, dic, abs_path)






