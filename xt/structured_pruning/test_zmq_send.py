import zmq  
  
def server():
    context = zmq.Context()  
    socket = context.socket(zmq.PULL)  
    socket.bind("tcp://*:5555")  
    
    message = socket.recv(b=False)  
    print("message from client:", message)  
    
    #  Send reply back to client  
    socket.send_string("World")


if __name__ == "__main__":
    server()
