import socket
import threading
import sys
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton

class CounterApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.PORT = 12345
        self.HOST = socket.gethostbyname(socket.gethostname())
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)    
    
    def handle_client(self, conn, addr):
        pass

    def start_server(self):
        self.sock.bind((self.HOST, self.PORT))
        self.sock.listen()
        while True:
            conn, addr = self.sock.accept()
            thread = threading.Thread(target=self.handle_client, args=(conn, addr))
            thread.start()        
            print(f"Connection from {addr} has been established!")

    def initUI(self):
        self.counter = 0  # Initial counter value
        layout = QVBoxLayout()        
        self.label = QLabel(f"Counter: {self.counter}")
        layout.addWidget(self.label)        
        self.button = QPushButton('Increment Counter')
        self.button.clicked.connect(self.increment_counter)
        layout.addWidget(self.button)
        self.setLayout(layout)
        self.setWindowTitle('Counter App')

    def increment_counter(self):
        self.counter += 1
        self.label.setText(f"Counter: {self.counter}")

def main():
    app = QApplication(sys.argv)
    window = CounterApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()


        
        