import sys
from PyQt5.QtCore import Qt, QCoreApplication
from PyQt5.QtGui import QImage, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget
from PIL import Image

class Drawer(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.setFixedSize(300, 300)
        self.setAttribute(Qt.WA_StaticContents)
        h = 300
        w = 300
        self.myPenWidth = 16
        self.myPenColor = Qt.white
        self.image = QImage(w, h, QImage.Format_RGB32)
        
        self.path = QPainterPath()
        self.clearImage()

    def setPenColor(self, newColor):
        self.myPenColor = newColor

    def setPenWidth(self, newWidth):
        self.myPenWidth = newWidth

    def clearImage(self):
        self.path = QPainterPath()
        self.image.fill(Qt.black)  
        self.update()

    def saveImage(self, fileName, fileFormat):
        self.image.save(fileName, fileFormat)
        foo = Image.open('image.png')
        foo = foo.resize((28,28), Image.Resampling.LANCZOS)
        foo.save('image.png', quality=95)
        QCoreApplication.instance().quit()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(event.rect(), self.image, self.rect())

    def mousePressEvent(self, event):
        self.path.moveTo(event.pos())

    def mouseMoveEvent(self, event):
        self.path.lineTo(event.pos())
        p = QPainter(self.image)
        p.setPen(QPen(self.myPenColor,
                      self.myPenWidth, Qt.SolidLine, Qt.RoundCap,
                      Qt.RoundJoin))
        p.drawPath(self.path)
        p.end()
        self.update()

def DisegnaTu():
    app = QApplication(sys.argv)
    w = QWidget()
    btnSave = QPushButton("Save image")
    btnClear = QPushButton("Clear")
    
    drawer = Drawer()

    w.setLayout(QVBoxLayout())
    w.layout().addWidget(btnSave)
    w.layout().addWidget(btnClear)
    w.layout().addWidget(drawer)

    btnSave.clicked.connect(lambda: drawer.saveImage("image.png", "JPG"))
    btnClear.clicked.connect(drawer.clearImage)    

    w.show()
    app.exec_()
    