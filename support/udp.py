import socket

def init_udp(ip=None, port=None):
    if ip is None:  
        UDP_IP = "localhost"
    else:
        UDP_IP = ip
    
    if port is None:
        UDP_PORT = 12345
    else:
        UDP_PORT = port
        
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM,)
    sock.bind((UDP_IP, UDP_PORT))
    print('socket binded')
    return sock

def init_tcp(ip=None, port=None):
    if ip is None:  
        TCP_IP = "localhost"
    else:
        TCP_IP = ip
    
    if port is None:
        TCP_PORT = 12345
    else:
        TCP_PORT = port
        
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((TCP_IP, TCP_PORT))
    print('socket binded')
    return sock