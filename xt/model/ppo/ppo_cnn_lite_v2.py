import os
import pickle
import struct
import sys
from copy import deepcopy
from time import sleep, time

import dill
import numpy as np
from xt.model.ppo.ppo_cnn import PpoCnn
from xt.model.ppo.py2c import send_state, recv_state

from zeus.common.util.register import Registers

from xt.model.tf_compat import tf, K, get_sess_graph
import zmq

@Registers.model
class PpoCnnLiteV2(PpoCnn):
    def __init__(self, model_info):
        tf.disable_eager_execution()
        super().__init__(model_info)
        self.interpreter = None
        self.keras_model_file_path = "explorer" + str(time()) + ".h5"
        self.predict_times = 0
        self.quantization = model_info.get("quantization", False)
        self.h5_file_prefix = model_info.get("h5_file_prefix", "explorer_tflite")

        self.init_interpret()
        self.pid = os.getpid()
        self.probs = None

        self.context = zmq.Context()
        self.send_socket = self.context.socket(zmq.REQ)
        # self.recv_socket = self.context.socket(zmq.PULL)
        self.k = 0
        self.init_send_socket = 99880
        self.init_recv_socket = 99980

        self.static_time = 0

        self.infer_time = 0
        self.infer_times = 0


    def create_model(self, model_info):
        model = super().create_model(model_info)
        # print("before PpoCnnLiteV2 model.summary() ==================== ")
        # model.summary()
        # print("after PpoCnnLiteV2 model.summary() ==================== ")

        return model

    def save_keras_model(self, ):
        # if os.path.exists(self.keras_model_file_path):
        #     os.remove(self.keras_model_file_path)
        self.keras_model_file_path = self.h5_file_prefix + str(time()) + ".h5"
        # print("================ save_keras_model ====================")
        # self.model.summary()
        # print("after save_keras_model ====================")

        with self.graph.as_default():
            K.set_session(self.sess)
            # -----------------------------------------------30ms
            tf.keras.models.save_model(self.model, self.keras_model_file_path)
            # -----------------------------------------------30ms
        # self.model.save(self.keras_model_file_path)
        return self.keras_model_file_path

    def post_model(self, tflite_model):
        self.interpreter = tf.lite.Interpreter(model_content=tflite_model)
        self.interpreter.allocate_tensors()

    def evaluate_model(self, state, user_bolt=False, re_data=None):
        # -------------------- 10-5s
        if user_bolt:
            pi_latent = [[re_data[0]]]
            out_v = [list(re_data[1:])]
            # with open("/home/xys/primary_xingtian/xingtian/record.txt", "a+") as f:
            #     f.write("bolt result ==== {}, {}\n".format(pi_latent, out_v))
                # print("bolt result ==== {}, {}".format(pi_latent, out_v))
            # print("bolt result ==== {}, {}".format(pi_latent, out_v))
        else:
            input_details = self.interpreter.get_input_details()
            output_details = self.interpreter.get_output_details()
            self.interpreter.resize_tensor_input(input_details[0]['index'], (len(state), 84, 84, 4))
            # self.interpreter.resize_tensor_input(output_details[0]['index'], (len(state), num_classes))
            self.interpreter.allocate_tensors()

            input_index = self.interpreter.get_input_details()[0]["index"]
            pi_latent_i = self.interpreter.get_output_details()[0]["index"]
            out_v_i = self.interpreter.get_output_details()[1]["index"]
            # -------------------- 10-5s

            # Pre-processing: add batch dimension and convert to float32 to match with
            # the model's input data format.
            # 2.--------------------------------------- 0.67ms
            start_evaluate1_time = time()
            state = state.astype(np.float32)
            # state = tf.convert_to_tensor(state, np.float32)
            self.interpreter.set_tensor(input_index, state)
            # Run inference.
            self.interpreter.invoke()
            pi_latent = self.interpreter.tensor(pi_latent_i)()
            out_v = self.interpreter.tensor(out_v_i)()


            # with open("/home/xys/primary_xingtian/xingtian/record.txt", "a+") as f:
            #     f.write("tflite result ==== {}, {}\n".format(pi_latent, out_v))
                # print("tflite result ==== {}, {}".format(pi_latent, out_v))
            # print("tflite result ==== {}, {}".format(pi_latent, out_v))
            # print(pi_latent)
            # print(out_v)
        # print("pi_latent == {}".format(pi_latent))
        # print("out_v ====={}".format(out_v))
        if len(pi_latent[0]) == 1:
            tmp = pi_latent
            pi_latent = out_v
            out_v = tmp
        # print("pi_latent ==== {}; out_v ====== {}".format(pi_latent, out_v))
        v_out = deepcopy(out_v)
        # 2.--------------------------------------- 0.67ms
        end_evaluate1_time = time()
        # print("self.sess.run2 time =============================== {}".format(end_evaluate1_time - start_evaluate1_time))

        # 3.-------------------- 0.45ms 有可能是tensorflow session启动所带来的开销
        start_evaluate1_time = time()
        # print("pi_latent============={}".format(pi_latent))
        # print("out_v============={}".format(out_v))
        actions = []
        logps = []
        for i in range(len(pi_latent)):
            action, logp = self.get_action([pi_latent[i]])
            actions.append(action[0])
            logps.append(logp[0])
            # print("logp[0] ============= {}".format(logp[0][0]))

        # print("action============={}".format(actions))
        # print("logps============={}".format(logps))

        end_evaluate1_time = time()
        # print("pid = {},  self.sess.run2 time =============================== {}".
        #       format(self.pid, end_evaluate1_time - start_evaluate1_time))
        # 3.-------------------- 0.45ms

        del pi_latent
        del out_v
        return actions, logps, v_out

    def get_action(self, pi_latent):
        # start_evaluate1_time = time()
        # if self.action_type == 'Categorical':
        #     dist_param = self.pi_latent_ph
        # elif self.action_type == 'DiagGaussian':
        #     # fixme: add input dependant log_std logic
        #     log_std = tf.get_variable('pi_logstd', shape=(1, self.action_dim), initializer=tf.zeros_initializer())
        #     dist_param = tf.concat([self.pi_latent_ph, self.pi_latent_ph * 0.0 + log_std], axis=-1)
        # else:
        #     raise NotImplementedError(
        #         'action type: {} not match any implemented distributions.'.format(self.action_type))
        # self.dist.init_by_param(dist_param)
        # self.action = self.dist.sample()
        # self.action_log_prob = self.dist.log_prob(self.action)

        # start_evaluate1_time = time()

        # tensorflow实现
        start_evaluate1_time = time()
        # with self.graph.as_default():
        #     feed_dict = {self.pi_latent_ph: pi_latent}
        #     action, logp = self.sess.run([self.action, self.action_log_prob], feed_dict)
        end_evaluate1_time = time()
        # print("self.sess.run1 time =============================== {}".format(end_evaluate1_time - start_evaluate1_time))

        # self.probs = self.softmax(pi_latent)
        # 不支持连续动作
        start_evaluate1_time = time()
        action = self.categorical_sample(pi_latent)
        np_logp = self.log_prob_np(action)
        # np_logp = np.array(np_logp)
        action = np.expand_dims(action, axis=0)
        np_logp = np.expand_dims(np_logp, axis=0)
        np_logp = np.expand_dims(np_logp, axis=0)
        end_evaluate1_time = time()
        # print("self.sess.run2 time =============================== {}".format(end_evaluate1_time - start_evaluate1_time))

        # print("tf.action = {}; tf.logp = {}; type(logp) = {}".format(action, logp, type(logp)))
        # print("np.action = {}; np.logp = {}; type(np_logp) = {}".format(action, np_logp, type(np_logp)))
        return action, np_logp

    def log_prob_np(self, action_index):
        # tf.one_hot
        x = np.eye(4)[action_index]

        # softmax_cross_entropy_with_logits_v2
        logp = -self.cross_entropy_error(x, np.array(self.probs[0]))
        # print("categorical_sample cross_entropy === {}".format(logp))
        return logp

    def cross_entropy_error(self, t, y):
        delta = 1e-7  # 添加一个微小值可以防止负无限大(np.log(0))的发生。
        return -np.sum(t * np.log(y + delta))

    def categorical_sample(self, logits):
        self.probs = self.softmax(logits)
        a = [i for i in range(len(logits[0]))]
        action_index = np.random.choice(a, p=self.probs[0])
        return action_index

    def softmax(self, logits):
        e_x = np.exp(logits)
        probs = e_x / np.sum(e_x, axis=-1, keepdims=True)
        return probs

    def predict(self, state, env_id=None):

        # print("len(state) ================= {}".format(len(state)))

        #bolt
        # if self.k == 0:
        #     print("env_id =========== {}".format(env_id))
        #     self.send_socket.connect("tcp://127.0.0.1:"+str(self.init_send_socket + env_id))
        #     # self.recv_socket.bind("tcp://127.0.0.1:" + str(self.init_recv_socket + env_id))
        #     self.k += 1
        start_evaluate1_time = time()
        # print(state.dtype)
        self.predict_times += 1
        # if self.predict_times % 40 == 0:
        #     print("self.predict_times ================== {}".format(self.predict_times))
        # pre = self.evaluate_model(state)
        # print(state)

        # here bolt
        # print(state.shape)


        # print("self.static_time ======== {}".format(self.static_time))
        # self.static_time = 0
        # print("write over===================")

        # with open("/home/xys/bolt/test/test/state_xt.txt", "w") as f:
        #     for x in np.nditer(state):
        #         # if x == 0.0:
        #         #     continue
        #         f.write(str(x))
        #         f.write("\n")
        #         self.static_time += 1
        #         if self.static_time > 28000:
        #             print(x)
        #     f.write("\n")
        #     f.write("\n")
        #     f.write("\n")
        #     f.write("\n")
        #     f.write(str(self.static_time))
        #     print("self.static_time ============== {}".format(self.static_time))

        # bolt
        # bolt_state = np.transpose(state, (0, 3, 1, 2))
        # # print(bolt_state.shape)
        # bolt_state = bolt_state.astype("float32")
        # data = bolt_state.tobytes()
        # self.send_socket.send(data)

        # send_state(state, self.send_socket)
        # cnt_array = np.where(state, 0, 1)
        # print(np.sum(cnt_array))
        # result = recv_state(self.send_socket)
        # print("before recv")
        # print("after recv")

        # print("over==============")
        # sleep(20)

        # bolt
        # message = self.send_socket.recv()
        # result = struct.unpack("<fffff", message)
        # pre = self.evaluate_model(state, True, result)

        # print("bolt result ====================== {}".format(result))

        pre = self.evaluate_model(state)
        # with open("/home/xys/primary_xingtian/xingtian/record.txt", "a+") as f:
        #     f.write("tflite pre =================== {}\n".format(pre))
        #     print("tflite pre =================== {}".format(pre))
        # print("tflite pre =================== {}".format(pre))

        # pre = self.evaluate_model(bolt_state, True, result)
        # with open("/home/xys/primary_xingtian/xingtian/record.txt", "a+") as f:
        #     f.write("bolt pre =================== {}\n".format(pre))
        #     print("bolt pre =================== {}".format(pre))

        # print("pre ====================== {}".format(pre))
        # sleep(20)
        end_evaluate1_time = time()
        # print("self.sess.run2 time =============================== {}".
        # format(end_evaluate1_time - start_evaluate1_time))
        print("pre ======================== {}".format(pre))
        return pre

    def get_convert_from_session(self):
        with self.graph.as_default():
            K.set_session(self.sess)
            converter = tf.lite.TFLiteConverter.from_session(self.sess, input_tensors=self.model.inputs,
                                                             output_tensors=self.model.outputs)
        return converter

    # version 2
    def set_weights(self, weights):
        if self.quantization:
            # print("=====================set_weights================================")
            serialized_model = weights
            tflite_model = dill.loads(serialized_model)

            # save_keras_model_filepath = self.save_keras_model()
            # converter = tf.lite.TFLiteConverter.from_keras_model_file(save_keras_model_filepath)
            # tflite_model = converter.convert()
            self.post_model(tflite_model)
        else:
            super().set_weights(weights)

    def get_weights(self):
        """Set weight with memory tensor."""
        if self.quantization:
            return None
        return self.model.get_weights()

    def init_interpret(self):
        save_keras_model_filepath = self.save_keras_model()
        # converter = tf.lite.TFLiteConverter.from_keras_model_file(self.keras_model_file_path, custom_objects={"loss": loss})
        # model1 = tf.keras.models.load_model(self.keras_model_file_path, custom_objects={"CustomModel": CustomModel})
        converter = tf.lite.TFLiteConverter.from_keras_model_file(save_keras_model_filepath)
        # converter.experimental_new_converter = True
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.SELECT_TF_OPS]
        # converter.allow_custom_ops = True
        tflite_model = converter.convert()
        self.post_model(tflite_model)
        print("no------------------------problem")

    def train(self, state, label):
        with self.graph.as_default():
            nbatch = state[0].shape[0]
            inds = np.arange(nbatch)
            loss_val = []
            for _ in range(self.num_sgd_iter):
                # Randomize the indexes
                np.random.shuffle(inds)
                # 0 to batch_size with batch_train_size step
                for start in range(0, nbatch, self._batch_size):
                    end = start + self._batch_size
                    mbinds = inds[start:end]
                    feed_dict = {self.state_ph: state[0][mbinds],
                                 self.behavior_action_ph: label[0][mbinds],
                                 self.old_logp_ph: label[1][mbinds],
                                 self.adv_ph: label[2][mbinds],
                                 self.old_v_ph: label[3][mbinds],
                                 self.target_v_ph: label[4][mbinds]}
                    ret_value = self.sess.run([self.train_op, self.loss], feed_dict)
                    loss_val.append(np.mean(ret_value[1]))

            return np.mean(loss_val)
