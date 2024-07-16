def get_button_ss():
    return """
            QPushButton {
                background-color: #4CAF50; /* Green */
                border: none;
                color: white;
                padding: 10px 24px;
                text-align: center;
                text-decoration: none;
                font-size: 16px;
                margin: 4px 2px;
                border-radius: 12px;
            }

            QPushButton:hover {
                background-color: #45a049; /* Darker Green */
            }

            QPushButton:pressed {
                background-color: #0E8A19; /* Pressed Green */
            }
        """


def get_label_ss():
    return """
            font-family: Arial;
            font-size: 24px;
            color: #333;
            font-weight: bold;
        """
