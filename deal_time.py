import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    infer_time_dicts = {}
    env_wait_dicts = {}
    file_env_step_dict = {}

    k = 0
    for i in range(5):
        if not str(i) in infer_time_dicts:
            infer_time_dicts.update({str(i): []})
            env_wait_dicts.update({str(i): []})
            file_env_step_dict.update({str(i): []})
        with open("lock_infer_time_" + str(i), "r") as f:
            while True:
                tmp = f.readline()
                tmp = tmp.rstrip("\n")
                if not tmp:
                    break
                k += 1
                if k > 500 and k < 600:
                    infer_time_dicts[str(i)].append(float(tmp))
                if k > 600: 
                    break

        with open("lock_env_wair_time_" + str(i), "r") as f:
            while True:
                tmp = f.readline()
                if not tmp:
                    break
                env_wait_dicts[str(i)].append(tmp)

        with open("file_env_step_time_" + str(i), "r") as f:
            while True:
                tmp = f.readline()
                tmp = tmp.rstrip("\n")
                if not tmp:
                    break
                k += 1
                if k > 500 and k < 1000:
                    file_env_step_dict[str(i)].append(float(tmp))
                if k > 1000: 
                    break
    x = [i for i in range(len(file_env_step_dict["0"]))]
    # print(infer_time_dicts["0"])
    plt.plot(x, file_env_step_dict["0"])
    plt.show()
    plt.savefig('loc_svg.svg',dpi=600) 