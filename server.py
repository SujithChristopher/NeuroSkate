import socket

# Define server address and port
server_address = ('localhost', 12345)

# Create a UDP socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Bind the socket to the server address
server_socket.bind(server_address)

print('Server is listening...')

# Continuous communication
while True:
    # Receive message from client
    data, client_address = server_socket.recvfrom(1024)
    
    # Print received message and client address
    print(f"Received from {client_address}: {data.decode('utf-8')}")

    # Echo the message back to the client
    server_socket.sendto(data, client_address)
