from . import usbrecord
import threading
import socket
import queue
audio_client_socket = None
server_send_queue = queue.Queue()


def StartSocketServer(IP, port, myWin):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ip_port = (IP, port)
    server.bind(ip_port)
    server.listen(5)
    sthread = threading.Thread(target=SocketBind, args=(
        server, myWin
    ))
    sthread.setDaemon(True)
    sthread.start()
    print("TCP Socket Server Started !")
    print("Bind at " + IP + ":" + str(port))


def SocketBind(ss, myWin):
    while True:
        conn, addr = ss.accept()
        latestconn = conn
        recvsthread = threading.Thread(target=SocketReceiving,
                                       args=(conn, myWin))
        recvsthread.setDaemon(True)
        recvsthread.start()


def SocketReceiving(incomingconn, myWin):
    global audio_client_socket
    try:
        while True:
            data = incomingconn.recv(1024)
            data = data.decode('utf-8')
            if(data == "hellohello"):
                print("update audio button")
                audio_client_socket = incomingconn
                myWin.enable_audio_button()
            else:
                datas = data.split("|")
                myWin.text_queue.put(datas[1])
                myWin.update()
                if(server_send_queue.qsize() > 0):
                    incomingconn.send(server_send_queue.get().encode("utf-8"))
                    if(server_send_queue.qsize() == 0):
                        print("queue is empty, reset first one flag")
                        usbrecord.is_not_the_first_one_flag = 0
                else:
                    usbrecord.is_not_the_first_one_flag = 0
    except ValueError:
        pass
