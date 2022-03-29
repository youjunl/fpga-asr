#from asyncio.windows_events import NULL
from queue import Queue
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from app.work import Ui_Form
import app.server as server
import app.usbrecord as usbrecord
from threading import Thread
from PyQt5 import QtCore
import os

_translate = QtCore.QCoreApplication.translate



class MyMainForm(QMainWindow, Ui_Form):
    text_queue = Queue()

    def __init__(self, parent=None):
        super(MyMainForm, self).__init__(parent)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.display)
        self.record_thread = []

    def appendtext(self, textcontent):
        self.textBrowser.append(textcontent+"\n")

    def closeEvent(self, event):
        event.accept()
        os._exit(0)

    def recording(self):
        self.rec = usbrecord.Recorder()
        self.rec.recording_symbol = 1
        self.rec.listen()

    def display(self):
        if(self.record_thread == []):
            self.pushButton.setText(_translate("Form", "Stop"))
            self.record_thread = Thread(target=self.recording, args=())
            self.record_thread.start()
        else:
            self.rec.recording_symbol = 0
            self.record_thread = []
            self.pushButton.setText(_translate("Form", "Audio"))

        return
        if(not server.audio_client_socket == None):
            server.audio_client_socket.send("please record".encode())
            self.textBrowser.setText('audio recording')
        else:
            self.textBrowser.setText('client not connected.')

    def appendtext_when_received(self, textcontent):
        self.textBrowser.appendPlainText(textcontent)
        self.update()

    def enable_audio_button(self):
        print("enable audio button")
        self.pushButton.setEnabled(True)
        self.pushButton.setText(_translate("Form", "Audio"))
        self.pushButton.update()
        self.update()

    def paintEvent(self, event):

        if(not self.text_queue.empty()):
            self.appendtext_when_received(self.text_queue.get())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MyMainForm()

    server.StartSocketServer("127.0.0.1", 10086, myWin)

    myWin.show()
    sys.exit(app.exec_())
