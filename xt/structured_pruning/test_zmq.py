from ast import main
import zmq  
  
def server():
    context = zmq.Context()  
    socket = context.socket(zmq.REP)  
    socket.bind("tcp://*:5555")  
    
    message = socket.recv()  
    print("message from client:", message)  
    
    #  Send reply back to client  
    socket.send("World")

def client():
    import zmq  
 
    context = zmq.Context()  
    print("Connecting to server...")
    socket = context.socket(zmq.REQ)  
    socket.connect ("tcp://localhost:5555")  
    
    socket.send ("Hello")  
    
    message = socket.recv()  
    print("Received reply: ", message)

if __name__ == "__main__":
    server()


