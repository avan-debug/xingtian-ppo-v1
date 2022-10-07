import struct
import time

import numpy as np
import json
import zmq
import pickle


def send_state(state, zmq_socket=None, port=None):
    # data = json.dumps(state.reshape(-1).tolist().copy())
    state.dtype = "int32"
    data = state.tobytes()
    if zmq_socket:
        # print("state.dtype = {}".format(state.dtype))
        # state.tobytes()
        zmq_socket.send(data)
    if port:
        context = zmq.Context()
        socket = context.socket(zmq.PUSH)
        socket.connect("tcp://127.0.0.1:99998")
        socket.send_string(data)
        socket.close()


def recv_state(zmq_socket=None, port=None):
    if zmq_socket:
        # msg = zmq_socket.recv_string()
        # data = json.loads(msg)
        message = zmq_socket.recv()
        data = struct.unpack("<fffff", message)
        return data
    if port:
        context = zmq.Context()
        socket = context.socket(zmq.PUSH)
        socket.connect("tcp://127.0.0.1:99998")
        msg = socket.recv_string()
        data = json.loads(msg)
        socket.close()
        return data


def main():
    context = zmq.Context()
    send_socket = context.socket(zmq.PUSH)
    recv_socket = context.socket(zmq.PULL)
    send_socket.connect("tcp://127.0.0.1:99992")
    recv_socket.bind("tcp://127.0.0.1:99994")
    data = np.zeros(shape=(4, 84, 84))
    s0 = time.time()
    # data = json.dumps(a.reshape(-1).tolist().copy())
    # socket.send(msg)
    for i in range(100):
        print("send state...{}...".format(i))
        send_state(data, send_socket)
        rdata = recv_state(recv_socket)
        print("recv data...{}... len {}".format(i, len(rdata)))
    send_socket.close()
    recv_socket.close()
    s1 = time.time()
    print(s1 - s0)
    return


if __name__ == '__main__':
    main()
