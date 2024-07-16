from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtGui import QPaintEvent
from PySide6.QtWidgets import *


class PyToggle(QCheckBox):
    def __ini__(
        self, width=60, bg_color="#777", circle_color="#DDD", active_color="00BCff"
    ):
        QCheckBox.__init__(self)

        self.setFixedSize(width, 28)
        self.setCursor(Qt.PointingHandCursor)

        self._bg_color = bg_color
        self._circle_color = circle_color
        self._active_color = active_color

        self.stateChanged.connect(self.debug)

    def debug(self):
        print(f"Status: {self.isChecked()}")

    def hitButton(self, pos: QPoint):
        return self.contentsRect().contains(pos)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        p.setPen(Qt.NoPen)

        rect = QRect(0, 0, self.width(), self.height())

        p.setBrush(QColor(self._bg_color))
        p.drawRoundedRect(0, 0, rect.width())
