import sys

import matplotlib
from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QMainWindow, QApplication, QLabel, QPushButton, QLineEdit, QHBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np

matplotlib.use('Qt5Agg')


class MplCanvas(FigureCanvas):
    def __init__(self, width, height, dpi=100, parent=None):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)


class QPlotView(QWidget):
    def __init__(self, width, height, dpi=100, show_toolbar=False, parent=None):
        super().__init__(parent)
        self.setLayout(QVBoxLayout())
        self.canvas = MplCanvas(width=width, height=height, dpi=dpi)
        self.fig = self.canvas.fig
        self.toolbar = NavigationToolbar(self.canvas, self)
        if show_toolbar:
            self.layout().addWidget(self.toolbar)
        self.layout().addWidget(self.canvas)


class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle('Dump Data Viewer')
        self.setFixedSize(QSize(1200, 700))

        # project selector
        self.layoutProjectSelector = QHBoxLayout()
        self.labelProjectName = QLabel()
        self.labelProjectName.setText('Project Name:')
        self.inputProjectName = QLineEdit()
        self.buttonOpenProject = QPushButton()
        self.buttonOpenProject.setText('Open')
        self.buttonOpenProject.clicked.connect(self.onButtonOpenProjectClicked)
        self.layoutProjectSelector.addWidget(self.labelProjectName)
        self.layoutProjectSelector.addWidget(self.inputProjectName)
        self.layoutProjectSelector.addWidget(self.buttonOpenProject)
        self.projectSelector = QWidget(self)
        self.projectSelector.setGeometry(0, 0, 600, 50)
        self.projectSelector.setLayout(self.layoutProjectSelector)

        # progress plot
        self.plotProgress = QPlotView(width=6, height=6, show_toolbar=True, parent=self)
        self.plotProgress.setGeometry(0, 50, 600, 600)

        # time selector
        self.layoutTimeSelector = QHBoxLayout()
        self.buttonBackward = QPushButton()
        self.buttonBackward.setText('<<')
        self.buttonPrevious = QPushButton()
        self.buttonPrevious.setText('<')
        self.buttonNext = QPushButton()
        self.buttonNext.setText('>')
        self.buttonForward = QPushButton()
        self.buttonForward.setText('>>')
        self.labelJumpTo = QLabel()
        self.labelJumpTo.setText('Jump to:')
        self.inputJumpTo = QLineEdit()
        self.buttonJump = QPushButton()
        self.buttonJump.setText('Jump')
        self.layoutTimeSelector.addWidget(self.buttonBackward)
        self.layoutTimeSelector.addWidget(self.buttonPrevious)
        self.layoutTimeSelector.addWidget(self.buttonNext)
        self.layoutTimeSelector.addWidget(self.buttonForward)
        self.layoutTimeSelector.addSpacing(50)
        self.layoutTimeSelector.addWidget(self.labelJumpTo)
        self.layoutTimeSelector.addWidget(self.inputJumpTo)
        self.layoutTimeSelector.addWidget(self.buttonJump)
        self.timeSelector = QWidget(self)
        self.timeSelector.setGeometry(0, 650, 600, 50)
        self.timeSelector.setLayout(self.layoutTimeSelector)

        # analysis plot
        self.plotAnalysis = QPlotView(width=6, height=7, show_toolbar=True, parent=self)
        self.plotAnalysis.setGeometry(600, 0, 600, 700)

    def onButtonOpenProjectClicked(self):
        print('hello')

    def openProject(self):
        project_name = self.inputProjectName.text()
        self.sf_real_time = np.load(f'output/{project_name}/sf_real_time.npy')



app = QApplication(sys.argv)
win = MainWindow()
win.show()
app.exec_()
