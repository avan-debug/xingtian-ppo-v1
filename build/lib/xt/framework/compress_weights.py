import logging
import time
from tkinter.messagebox import NO
import tracemalloc
from multiprocessing import Process, Queue

import subprocess
import dill
import os
from setproctitle import setproctitle

import tensorflow as tf
tf.compat.v1.enable_eager_execution()
# from tensorflow_model_optimization.python.core.sparsity.keras.prune import prune_low_magnitude

from xt.model.impala.impala_cnn_lite import CustomModel
# from xt.structured_pruning.src.test.pruning_function_test import pruning_model

run_time = time.time()

from xt.framework import global_var

import zmq

class CompressWeights:
    def __init__(self, **kwargs):
        shared_queue = kwargs.get("shared_queue")
        self.raw_weights_queue = shared_queue[0]  # type:Queue
        self.compress_weights_queue = shared_queue[1]  # type:Queue
        self.compress_tool = None
        self.file_lock = global_var.get_value("lock")



    def register_weights_process_function(self, func):
        self.compress_tool = func

    def task_loop(self):
        # tracemalloc.start()
        # print("start tracemalloc===================")
        setproctitle("xt_compress")
        os.environ["CUDA_VISIBLE_DEVICES"] = str("-1")
        while True:
            # print("raw_weights_queue.length==={}".format(self.raw_weights_queue.qsize()))
            # print("compress_weights_queue.length==={}".format(self.compress_weights_queue.qsize()))

            model_file_path = self.raw_weights_queue.get()
            if not self.raw_weights_queue.empty():
                os.remove(model_file_path)
                continue
            start_compass = time.time()

            
            # print("before self.lock.writer_lock.acquire()========================{}".format(self.file_lock))
            serialize_tflite_model = self.compress_tool(model_file_path, self.file_lock)
            # print("after self.lock.writer_lock.release()==========================")
            # print("compass weight lock======================= {}".format(self.file_lock))
            end_compass = time.time()
            # print("compass time ================= {}".format(end_compass - start_compass))
            while not self.compress_weights_queue.empty():
                self.compress_weights_queue.get()
            self.compress_weights_queue.put(serialize_tflite_model)
            # print("compass weight")


    def start(self):
        Process(target=self.task_loop).start()


def serialize_tflite(model_file_path):
    # print("tf.version ============== {}".format(tf.__version__))
    model1 = tf.keras.models.load_model(model_file_path,  custom_objects={"CustomModel": CustomModel})
    # print("model1 =================== {}")
    # model1.summary()
    # pruned_model = pruning_model(model1)
    # pruned_model.summary()
    converter = tf.lite.TFLiteConverter.from_keras_model(model1)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.inference_input_type = tf.int8
    tflite_model = converter.convert()
    # with open('ppo_cnn.tflite', 'wb') as f:
    #     f.write(tflite_model)

    # print("over ====================================")

    serialize_tflite_model = dill.dumps(tflite_model)
    os.remove(model_file_path)
    return serialize_tflite_model


def serialize_bolt(model_file_path, lock=None):
    # print("start  serialize_bolt==============================")
    model1 = tf.keras.models.load_model(model_file_path, custom_objects={"CustomModel": CustomModel})
    # print("model1 =================== {}")
    # model1.summary()
    # pruned_model = pruning_model(model1)
    # pruned_model.summary()
    converter = tf.lite.TFLiteConverter.from_keras_model(model1)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    # print("before compasss weight===========================")
    with open('/home/xys/bolt/test/ppo_cnn.tflite', 'wb') as f:
        f.write(tflite_model)
    lock.acquire()
    p = subprocess.Popen(
        '/home/xys/bolt/install_linux-x86_64_avx2/tools/X2bolt -d /home/xys/bolt/test'
        ' -m ppo_cnn -i FP32', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in p.stdout.readlines():
        print(line)
        break
    lock.release()

    # print("===================subprocess ok ===============================")

    serialize_tflite_model = dill.dumps(tflite_model)
    os.remove(model_file_path)
    return serialize_tflite_model


def empty_weights_proc_func(weights):
    return weights
