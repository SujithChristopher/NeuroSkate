import toml
from run_gui import MainWindow
from run_no_gui import Oskar
from PySide6.QtWidgets import QApplication
import sys

_toml_path = "settings.toml"
settings = toml.load(_toml_path)

if settings['display']['display']:

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()

else:
    Oskar()