import socket

# Define server address and port
server_address = ('localhost', 12345)

# Create a UDP socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Continuous communication
while True:

    # Send message to server
    client_socket.sendto('message'.encode('utf-8'), server_address)

    response, server = client_socket.recvfrom(1024)
    print(f"Response from server: {response.decode('utf-8')}")
