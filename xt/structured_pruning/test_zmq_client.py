import zmq  
  


def client():
    import zmq  
 
    context = zmq.Context()  
    print("Connecting to server...")
    socket = context.socket(zmq.PUSH)  
    socket.connect ("tcp://localhost:5555")  
    
    socket.send_string("Hello")  
    
    message = socket.recv()  
    print("Received reply: ", message)

if __name__ == "__main__":
    client()


