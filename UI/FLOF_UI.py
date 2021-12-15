# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'FLOF_UI.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1920, 980)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.filkom_img = QtWidgets.QLabel(self.centralwidget)
        self.filkom_img.setGeometry(QtCore.QRect(30, 20, 291, 91))
        self.filkom_img.setText("")
        self.filkom_img.setPixmap(QtGui.QPixmap("../assets/filkom.png"))
        self.filkom_img.setScaledContents(True)
        self.filkom_img.setObjectName("filkom_img")
        self.hint_img = QtWidgets.QLabel(self.centralwidget)
        self.hint_img.setGeometry(QtCore.QRect(720, 860, 561, 91))
        self.hint_img.setText("")
        self.hint_img.setPixmap(QtGui.QPixmap("../assets/hint.png"))
        self.hint_img.setScaledContents(True)
        self.hint_img.setObjectName("hint_img")
        self.camera = QtWidgets.QLabel(self.centralwidget)
        self.camera.setGeometry(QtCore.QRect(1620, 20, 271, 231))
        self.camera.setStyleSheet(" border: 5px solid black;")
        self.camera.setText("")
        self.camera.setObjectName("camera")
        self.camera_btn = QtWidgets.QPushButton(self.centralwidget)
        self.camera_btn.setGeometry(QtCore.QRect(1620, 270, 271, 51))
        self.camera_btn.setObjectName("camera_btn")
        self.title = QtWidgets.QLabel(self.centralwidget)
        self.title.setGeometry(QtCore.QRect(630, 30, 511, 141))
        font = QtGui.QFont()
        font.setPointSize(39)
        self.title.setFont(font)
        self.title.setTextFormat(QtCore.Qt.PlainText)
        self.title.setScaledContents(True)
        self.title.setAlignment(QtCore.Qt.AlignCenter)
        self.title.setObjectName("title")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(40, 340, 1861, 301))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(20)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.food_5 = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.food_5.setAlignment(QtCore.Qt.AlignCenter)
        self.food_5.setObjectName("food_5")
        self.horizontalLayout.addWidget(self.food_5)
        self.food_6 = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.food_6.setAlignment(QtCore.Qt.AlignCenter)
        self.food_6.setObjectName("food_6")
        self.horizontalLayout.addWidget(self.food_6)
        self.food_4 = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.food_4.setAlignment(QtCore.Qt.AlignCenter)
        self.food_4.setObjectName("food_4")
        self.horizontalLayout.addWidget(self.food_4)
        self.food_7 = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.food_7.setAlignment(QtCore.Qt.AlignCenter)
        self.food_7.setObjectName("food_7")
        self.horizontalLayout.addWidget(self.food_7)
        self.food_8 = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.food_8.setAlignment(QtCore.Qt.AlignCenter)
        self.food_8.setObjectName("food_8")
        self.horizontalLayout.addWidget(self.food_8)
        self.food_9 = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.food_9.setAlignment(QtCore.Qt.AlignCenter)
        self.food_9.setObjectName("food_9")
        self.horizontalLayout.addWidget(self.food_9)
        self.pick_img = QtWidgets.QLabel(self.centralwidget)
        self.pick_img.setGeometry(QtCore.QRect(40, 650, 1851, 61))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.pick_img.setFont(font)
        self.pick_img.setAlignment(QtCore.Qt.AlignCenter)
        self.pick_img.setObjectName("pick_img")
        self.food_picked = QtWidgets.QLabel(self.centralwidget)
        self.food_picked.setGeometry(QtCore.QRect(50, 740, 591, 71))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.food_picked.setFont(font)
        self.food_picked.setObjectName("food_picked")
        self.next = QtWidgets.QPushButton(self.centralwidget)
        self.next.setGeometry(QtCore.QRect(1320, 270, 271, 51))
        self.next.setObjectName("next")
        self.prev = QtWidgets.QPushButton(self.centralwidget)
        self.prev.setGeometry(QtCore.QRect(1030, 270, 271, 51))
        self.prev.setObjectName("prev")
        self.message = QtWidgets.QLabel(self.centralwidget)
        self.message.setGeometry(QtCore.QRect(40, 250, 511, 71))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.message.setFont(font)
        self.message.setTextFormat(QtCore.Qt.PlainText)
        self.message.setScaledContents(True)
        self.message.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.message.setObjectName("message")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.camera_btn.setText(_translate("MainWindow", "Start"))
        self.title.setText(_translate("MainWindow", "SMART CANTEEN"))
        self.food_5.setText(_translate("MainWindow", "Mie Ayam"))
        self.food_6.setText(_translate("MainWindow", "Bakso Sapi"))
        self.food_4.setText(_translate("MainWindow", "Soto Ayam"))
        self.food_7.setText(_translate("MainWindow", "Nasi Kuli"))
        self.food_8.setText(_translate("MainWindow", "Bihun Goreng"))
        self.food_9.setText(_translate("MainWindow", "Pecel Ayam"))
        self.pick_img.setText(_translate("MainWindow", "Pilih"))
        self.food_picked.setText(_translate("MainWindow", "Makanan Anda : Nasi Kuli"))
        self.next.setText(_translate("MainWindow", "Next"))
        self.prev.setText(_translate("MainWindow", "Prev"))
        self.message.setText(_translate("MainWindow", "SMART CANTEEN"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

