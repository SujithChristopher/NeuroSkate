import socket

# Define the host and port to listen on
HOST = '127.0.0.1'
PORT = 12345

# Create a TCP socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
    # Bind the socket to the host and port
    server_socket.bind((HOST, PORT))
    
    # Start listening for incoming connections
    server_socket.listen()

    print(f"Server listening on {HOST}:{PORT}")

    # Accept incoming connections
    client_socket, client_address = server_socket.accept()

    print(f"Connected to {client_address}")

    # Continuously send and receive data
    while True:
        # Receive data from the client
        data = client_socket.recv(1024)

        # If no data is received, break the loop
        if not data:
            break

        # Print received data
        print(f"Received data from client: {data.decode()}")

        # Send a response back to the client
        response = input("Enter response: ")
        client_socket.sendall(response.encode())

    # Close the client socket
    client_socket.close()
