import datetime
import os.path
import re
import threading
import time
import setproctitle
import psutil
from tensorboardX import GlobalSummaryWriter
import subprocess as subs
import multiprocessing as multi
import shlex
import logging
import argparse
import subprocess as subp
from ruamel import yaml
import zmq
import json

parser = argparse.ArgumentParser(description='monitor')
parser.add_argument('--logdir', type=str, default="./logs/test", help="where logs stored")
parser.add_argument('--name', '-n', type=str, default='140', help="name of log")
parser.add_argument('--file', '-f', type=str, default='', help="file of log config")
parser.add_argument('--remote', action="store_true")
parser.add_argument('--auto', action="store_true")
parser.add_argument('--device', '-d', type=str, default='')
parser.add_argument('--no_pid', action="store_true")
args = parser.parse_args()

start_port = "5532"
trans_port = "5533"


class Comm:
    def __init__(self, *args, **kwargs):

        try:
            self.ip = args[0]
        except IndexError:
            self.ip = kwargs.get("ip", "127.0.0.1")

        try:
            self.port = args[1]
        except IndexError:
            self.port = kwargs.get("port", "5558")

        try:
            self.type = args[2]
        except IndexError:
            self.type = kwargs.get("type", "PUSH")
        TYPE = {
            "PUSH": zmq.PUSH,
            "PULL": zmq.PULL
        }
        self.type = TYPE.get(self.type)
        self.socket = None

    def build(self, type='conn'):
        context = zmq.Context()
        socket = context.socket(self.type)
        if self.type == zmq.PUSH:
            if type == "bind":
                socket.bind("tcp://*:{}".format(self.port))
            else:
                socket.connect("tcp://{}:{}".format(self.ip, self.port))
        else:
            if type == "bind":
                socket.bind("tcp://*:{}".format(self.port))
            else:
                socket.connect("tcp://{}:{}".format(self.ip, self.port))

        self.socket = socket
        return self

    @staticmethod
    def mogrify(topic, msg):
        """ json encode the message and prepend the topic """
        return topic + ' ' + json.dumps(msg)

    @staticmethod
    def demogrify(topicmsg):
        """ Inverse of mogrify() """
        json0 = topicmsg.find('{')
        topic = topicmsg[0:json0].strip()
        msg = json.loads(topicmsg[json0:])
        return topic, msg

    def send(self, topic, msg: dict):
        self.socket.send(Comm.mogrify(topic, msg).encode())

    def recv(self):
        topic, msg = Comm.demogrify(self.socket.recv().decode())
        return topic, msg

    def getsocket(self):
        return self.socket


class Monitor:
    _dict = dict()

    def __init__(self, *args, **kwargs):
        self.writer = kwargs.get('writer', GlobalSummaryWriter())
        self.cmd = kwargs.get('cmd', 'ls')
        self.interval = kwargs.get('interval', 1)
        self.event = kwargs.get('event', [['cycles']])
        self.device = kwargs.get('device', 'cpu')
        self.pids = kwargs.get('pids', [])
        self.sleep = kwargs.get('sleep', 1)
        self.time = kwargs.get('time', 30)
        self.cores = kwargs.get('cores', '[0-31]')
        self.name = kwargs.get("name", "localhost")
        self.net_monitor_port = kwargs.get("net_monitor_port", "6954")
        # self.writer=GlobalSummaryWriter()

    def start(self):
        setproctitle.setproctitle("monitor_xdz")
        self.pids = self._dict['pids_of']('xt_explorer') if not self.pids else self.pids
        self.cmd = self.cmd_split(cmd=self.cmd, event=self.event, pids=self.pids, sleep=self.sleep, time=self.time,
                                  cores=self.cores)
        while True:
            for cmd in self.cmd:
                print(cmd)
                logging.info(cmd)
                with subs.Popen(shlex.split(cmd), stdin=subs.PIPE, stdout=subs.PIPE, stderr=subs.PIPE) as proc:
                    time.sleep(self.sleep)
                    raw_txt = str(proc.stdout.read() + proc.stderr.read(), 'utf-8')

                    # data process
                    label, scalars = self.__process(self.device, raw_txt)
                    if len(label) == 0:
                        continue
                    for l, d in zip(label, scalars):
                        try:
                            self.writer.add_scalar(self.device + '/' + l, d)
                            # self.writer.add_text("text/{}/{}".format(self.device, l), raw_txt)
                            # print(l, d)
                        except Exception as e:
                            print("start error {}".format(e))

    def cmd_split(self, cmd, event, pids, sleep, time, cores):
        cmd_list = []
        for e in event:
            cmd_list.append(cmd(event=e, pids=pids, interval=sleep, time=time, cores=cores))
        return cmd_list

    @staticmethod
    def mogrify(topic, msg):
        """ json encode the message and prepend the topic """
        return topic + ' ' + json.dumps(msg)

    @staticmethod
    def demogrify(topicmsg):
        """ Inverse of mogrify() """
        json0 = topicmsg.find('{')
        topic = topicmsg[0:json0].strip()
        msg = json.loads(topicmsg[json0:])
        return topic, msg

    def remote_monitor(self, *args, **kwargs):
        setproctitle.setproctitle("monitor_xdz")
        # net monitor data receive
        # net_receiver = Comm("127.0.0.1", self.net_monitor_port, "PULL").build()

        try:
            ip = args[0]
        except IndexError:
            ip = kwargs.get("ip", "192.168.1.138")

        try:
            port = args[1]
        except IndexError:
            port = kwargs.get("port", "5558")

        remote_record = Comm("ip", port, "PULL").build("bind")
        while True:
            topic, data = remote_record.recv()
            print('******************remote normal.************************')
            if len(data) == 0:
                continue
            for l, d in data.items():
                if l == "text":
                    # self.writer.add_text("text/" + topic + '/' + l, d)
                    pass
                else:
                    self.writer.add_scalar(topic + '/' + l, d)

    def remote_start(self, *args, **kwargs):
        setproctitle.setproctitle("monitor_xdz")
        try:
            ip = args[0]
        except IndexError:
            ip = kwargs.get("ip", "127.0.0.1")

        try:
            port = args[1]
        except IndexError:
            port = kwargs.get("port", "5558")
        print("port:{}".format(port))
        record = Comm(ip, port, "PUSH").build()

        self.pids = self._dict['pids_of']('xt_explorer') if not self.pids else self.pids
        # self.cmd = self.cmd(event=self.event, pids=self.pids, interval=self.sleep, time=self.time, cores=self.cores)
        self.cmd = self.cmd_split(cmd=self.cmd, event=self.event, pids=self.pids, sleep=self.sleep, time=self.time,
                                  cores=self.cores)
        while True:
            for cmd in self.cmd:
                print(cmd)
                logging.info(cmd)
                with subs.Popen(shlex.split(cmd), stdin=subs.PIPE, stdout=subs.PIPE, stderr=subs.PIPE) as proc:
                    time.sleep(self.sleep)
                    raw_txt = str(proc.stdout.read() + proc.stderr.read(), 'utf-8')
                    # print(raw_txt)

                    # data process

                    label, scalars = self.__process(self.device, raw_txt)
                    if len(label) == 0:
                        continue
                    # for l, d in zip(label, scalars):
                    #     self.writer.add_scalar(self.device + '/' + l, d)
                    #     self.writer.add_text("text/{}/{}".format(self.device, l), raw_txt)
                    for l, d in zip(label, scalars):
                        name = self.name
                        raw_data = {self.device + '/' + l: d, "text": raw_txt}
                        record.send(name, raw_data)

    def monitor_net__(self, file="./net.log"):
        cmd = "ifstat -n -i ens4f0 -t &>{}".format(file)
        subs.Popen(shlex.split(cmd), stdin=subs.PIPE, stdout=subs.PIPE, stderr=subs.PIPE)

    def monitor_net(self, port):
        setproctitle.setproctitle("monitor_xdz")
        # net_comm = Comm("127.0.0.1", port, "PUSH").build()
        cmd = "ifstat -n -i ens4f0 -t 2 1"
        while True:
            with subs.Popen(shlex.split(cmd), stdin=subs.PIPE, stdout=subs.PIPE, stderr=subs.PIPE) as proc:
                time.sleep(2)
                raw_txt = str(proc.stdout.read() + proc.stderr.read(), 'utf-8')
                raw_data = raw_txt.split("\n")[2].split()
                print(raw_data)
                try:
                    nin = float(raw_data[1])
                    nout = float(raw_data[2])
                    # net_comm.send("ens4f0", {self.device + '/' + "in_KB_s)": nin, "text": raw_txt})
                    # net_comm.send("ens4f0", {self.device + '/' + "out_KB_s)": nout, "text": raw_txt})
                    self.writer.add_scalar(self.device + '/' + "in_KB_s)", nin)
                    self.writer.add_scalar(self.device + '/' + "out_KB_s)", nout)
                    # self.writer.add_text("text/{}/{}".format(self.device, "netio"), raw_txt)
                except Exception as e:
                    print("monitor net error", e)
                    pass

    @staticmethod
    def register(param):
        if not callable(param):
            raise Exception("To Registry must be callable, Got: {}.".format(param))

        register_name = param.__name__
        if register_name in Monitor._dict:
            logging.warning("Key:{} is registered".format(register_name))

        Monitor._dict[register_name] = param
        return param

    @staticmethod
    def func(name):
        try:
            return Monitor._dict[name]
        except KeyError as e:
            return None

    def __process(self, name, data):
        try:
            ee = []
            for e in self.event:
                ee.extend(e)
            return self._dict[name](data, ee)
        except KeyError as e:
            print('{} not exists.\nerror {}'.format(name, e))
            return [], []

    def __getitem__(self, item):
        try:
            return Monitor._dict[item]
        except KeyError as e:
            print("{} not exist".format(item))


@Monitor.register
def cpu(raw_txt, event):
    # cpu log data process
    raw_data = [e.replace('#', '').split('SEP') for e in raw_txt.split('\n')]
    data = []
    for e in raw_data:
        k = set(e).intersection(set(event))
        if k:
            bg = e.index(list(k)[0])
            data.append([e[bg], *e[-2:]])

    label = []
    scalars = []
    for d in data:
        u = d[-1]
        # print(d)
        try:
            s = float(d[1])
        except Exception as e:
            print("Error.{}.raw:{}".format(e, d))
            s = 0
            pass
        if 'M' in d[-1]:
            u = u.replace('M', 'K')
            s = s * 1000
        label.append(d[0] + "({})".format(u))
        scalars.append(s)
    # print(label, scalars)
    # scalars = [eval(e) for e in scalars]
    return label, scalars


@Monitor.register
def gpu(raw_txt, event):
    # gpu log data process
    raw_data = raw_txt.split('\n')
    label = shlex.split(raw_data[0].replace('#', ''))
    unit = shlex.split(raw_data[1].replace('#', ''))
    label = [l + "({})".format(u) for l, u in zip(label, unit)]
    scalars = shlex.split(raw_data[2])
    scalars = [eval(e) for e in scalars]
    return label, scalars


# @Monitor.register
# def mem(raw_txt, event):
#     # gpu log data process
#     raw_data = re.split("TIME.*?\n", raw_txt)[1:]
#     raw_data = [e.split('\n')[:-1] for e in raw_data]
#     data = {}
#     cnt = 0
#     for l in raw_data:
#         cnt += 1
#         for e, f in zip(shlex.split(l[0]), shlex.split(l[1])):
#             if e == 'CORE':
#                 data[e] = f
#             else:
#                 data[e] = eval(f.replace('k', '000')) if e not in data else data[e] + eval(f.replace('k', '000'))
#     for k in data.keys():
#         data[k] = data[k] if isinstance(k, str) else data[k] / cnt
#     try:
#         data.pop('CORE')
#     except KeyError as e:
#         return [], []
#     label = list(data.keys())
#     scalars = list(data.values())
#     # print(label, scalars)
#     return label, scalars
@Monitor.register
def mem(raw_txt, event):
    # gpu log data process
    raw_data = re.split("TIME.*?\n", raw_txt)[1:]
    raw_data = [e.split('\n')[:-1] for e in raw_data]
    labels = []
    scalars = []
    for l in raw_data:
        for e, f in zip(shlex.split(l[0]), shlex.split(l[1])):
            if e == 'CORE':
                # label.append(e)
                # scalars.append(f)
                pass
            else:
                labels.append(e)
                scalars.append(float(f.replace('k', '000')))
    return labels, scalars


@Monitor.register
def pids_of(name):
    raw_txt = subs.getoutput("pgrep {}".format(name))
    res = shlex.split(raw_txt)
    return res


session = {
    'cpu': lambda *args, **kwargs: "perf stat -x SEP -e " + ("{}," * len(kwargs["event"])).format(*kwargs["event"])[
                                                            :-1] + " -p "
                                   + ("{}," * len(kwargs['pids'])).format(*kwargs['pids'])[
                                     :-1] + ' sleep ' + "{}".format(kwargs["interval"]),
    'gpu': lambda *args, **kwargs: "nvidia-smi dmon -s muct -c 1",
    'mem': lambda *args, **kwargs: "pqos -r -t {} -m all:{}".format(kwargs['time'], kwargs['cores'])
}

event = [["task-clock", "cycles", "instructions", "branches", "branch-misses"],
         ["task-clock", "context-switches", "cpu-migrations", "page-faults"],
         ["task-clock", "L1-dcache-loads", "L1-dcache-load-misses"],
         ["task-clock", "LLC-loads", "LLC-load-misses", "dTLB-loads", "dTLB-load-misses"]]


def main():
    global start
    name = args.name
    if args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            raw_cfg = f.read()
            cfg = yaml.load(raw_cfg, Loader=yaml.Loader)
        name = "monitor_" + cfg['benchmark']['id']
        logdir = os.path.join(cfg['benchmark']['archive_root'],
                              name + datetime.datetime.now().strftime("-%Y%m%d-%H%M%S"))
    else:
        logdir = os.path.join(args.logdir, name + datetime.datetime.now().strftime("-%Y%m%d-%H%M%S"))

    writer = GlobalSummaryWriter(logdir=logdir)
    mon_cpu = Monitor(device='cpu', writer=writer, cmd=session['cpu'], event=event)
    mon_gpu = Monitor(device='gpu', writer=writer, cmd=session['gpu'], interval=2)
    mon_mem = Monitor(device='mem', writer=writer, cmd=session['mem'], time=30, cores='[0-79]')

    mon_cpu_proc = multi.Process(target=mon_cpu.start)
    mon_gpu_proc = multi.Process(target=mon_gpu.start)
    mon_mem_proc = multi.Process(target=mon_mem.start)

    run_proc_list = [mon_mem_proc, mon_cpu_proc, mon_gpu_proc]
    # run_proc_list = []
    while True:
        print("waiting")
        if os.path.exists("./run.log"):
            res = subp.getoutput("cat run.log | grep Steps")
            # print(res)
            if res.find("Steps") >= 0:
                print("start local.")
                mon_cpu_proc.start()
                mon_gpu_proc.start()
                mon_mem_proc.start()

                if args.device:
                    print("remote start.")
                    start = Comm("localhost", start_port, "PUSH").build("bind")

                    mon_rem = Monitor(device='s138', writer=writer, net_monitor_port=trans_port)
                    mon_rem_proc = multi.Process(target=mon_rem.remote_monitor, args=("", trans_port,))
                    mon_rem_proc.start()
                    run_proc_list.append(mon_rem_proc)

                    mon_net = Monitor(device='net', writer=writer, cmd="read file tmp.")
                    mon_net_proc = multi.Process(target=mon_net.monitor_net, args=(trans_port,))
                    mon_net_proc.start()
                    run_proc_list.append(mon_net_proc)
                    start.send("info", {"info": "start"})
                    # run_proc_list.append(mon_net_proc__)
                break  # jump outier
        time.sleep(2)
    logging.info("start monitor.")
    try:
        while True:
            if not args.no_pid:
                pid_len = len(Monitor.func('pids_of')('xt_explorer'))
                if pid_len == 0:
                    raise RuntimeError("no pid exit.")

            if os.path.exists("./run.log"):
                res = subp.getoutput("cat run.log | grep Finished")
                print(res)
                if res.find("Finished") >= 0:
                    print("monitor close!")
                    raise RuntimeError("monitor close.")

                else:
                    continue
            time.sleep(10)
        # while True:
        #     # res = subp.getoutput("pgrep xt_explorer")
        #     print(res)
        #     pass
        for proc in run_proc_list:
            proc.join()
    except (KeyboardInterrupt, RuntimeError) as e:
        # print(e)
        print("monitor close.")
        # start.send("info", {"info": "end"})
        # print("remote close.")
        for proc in run_proc_list:
            proc.terminate()


def remote_main():
    name = args.name
    logdir = args.logdir

    writer = None
    mon_cpu = Monitor(name=name, device='cpu', writer=writer, cmd=session['cpu'], event=event)
    mon_gpu = Monitor(name=name, device='gpu', writer=writer, cmd=session['gpu'], interval=2)
    mon_mem = Monitor(name=name, device='mem', writer=writer, cmd=session['mem'], time=30, cores='[0-31]')

    mon_cpu_proc = multi.Process(target=mon_cpu.remote_start, args=("192.168.1.128", trans_port,))
    mon_gpu_proc = multi.Process(target=mon_gpu.remote_start, args=("192.168.1.128", trans_port,))
    mon_mem_proc = multi.Process(target=mon_mem.remote_start, args=("192.168.1.128", trans_port,))

    start = Comm("192.168.1.128", start_port, "PULL").build()

    while True:
        print("waiting")
        _, flag = start.recv()
        if flag["info"] == 'start':
            mon_cpu_proc.start()
            mon_gpu_proc.start()
            mon_mem_proc.start()

            break
        time.sleep(2)
    logging.info("start monitor.")
    print("start remote monitor.")
    try:
        while True:
            print("monitoring!")
            _, flag = start.recv()
            if flag["info"] == 'end':
                raise RuntimeError
            pid_len = len(Monitor.func('pids_of')('xt_explorer'))
            if pid_len == 0:
                raise RuntimeError
            time.sleep(10)
    except (KeyboardInterrupt, RuntimeError) as e:
        print(e)
        mon_cpu_proc.terminate()
        mon_gpu_proc.terminate()
        mon_mem_proc.terminate()
        # mon_gpu_proc.kill()
        # mon_mem_proc.kill()


if __name__ == '__main__':
    print(args.remote, " yes")
    if not args.remote:
        print("local")
        main()
    else:
        print("remote")
        remote_main()
    # Monitor.show()
    # mon_mem = Monitor(device='mem', cmd=session['mem'], time=1)
    # mon_mem.start()
