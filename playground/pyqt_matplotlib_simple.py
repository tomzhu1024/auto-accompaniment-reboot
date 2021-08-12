import random
import sys

import matplotlib
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

matplotlib.use('Qt5Agg')


class MplCanvas(FigureCanvas):
    def __init__(self, width=5, height=4, dpi=100, parent=None):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)


class QPlotView(QtWidgets.QWidget):
    def __init__(self, width=5, height=4, dpi=100, show_toolbar=False, parent=None):
        super().__init__(parent)
        self.setLayout(QtWidgets.QVBoxLayout())
        self.canvas = MplCanvas(width=width, height=height, dpi=dpi)
        self.fig = self.canvas.fig
        self.toolbar = NavigationToolbar(self.canvas, self)
        if show_toolbar:
            self.layout().addWidget(self.toolbar)
        self.layout().addWidget(self.canvas)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle('Dump Data Viewer')
        self.setFixedSize(1600, 800)
        plot = QPlotView(width=8, height=6, dpi=100, show_toolbar=True, parent=self)
        plot.setGeometry(0, 0, 800, 600)
        axes = plot.fig.add_subplot(111)
        axes.plot(list(range(1000)), [random.random() for i in range(1000)])
        axes.clear()
        axes.plot(list(range(1000)), [random.random() for i in range(1000)])


app = QtWidgets.QApplication(sys.argv)
win = MainWindow()
win.show()
app.exec_()
