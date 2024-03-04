import socket

# Define the server's IP address and port
SERVER_HOST = '127.0.0.1'
SERVER_PORT = 12345

# Create a TCP socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
    # Connect to the server
    client_socket.connect((SERVER_HOST, SERVER_PORT))

    # Continuously send and receive data
    while True:
        # Send data to the server
        message = input("Enter message: ")
        client_socket.sendall(message.encode())

        # Receive data from the server
        data = client_socket.recv(1024)

        # If no data is received, break the loop
        if not data:
            break

        # Print received data
        print(f"Received data from server: {data.decode()}")
