import socket

def init_udp(ip=None, port=None):
    if ip is None:  
        UDP_IP = "127.0.0.1"
    else:
        UDP_IP = ip
    
    if port is None:
        UDP_PORT = 8000
    else:
        UDP_PORT = port
        
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    return sock

