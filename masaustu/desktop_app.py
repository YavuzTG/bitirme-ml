import os
import sys

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication


def main() -> None:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    from app import MainWindow

    qt_app = QApplication(sys.argv)
    qt_app.setFont(QFont("Segoe UI", 10))
    window = MainWindow()
    window.show()
    sys.exit(qt_app.exec_())


if __name__ == "__main__":
    main()
