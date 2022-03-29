import numpy as np
from utils.data_layer import AudioDataLayer
from frame import FrameASR, FrameVAD
from torch.utils.data import DataLoader
import socket
import wave
import os

backup_dir = 'audio'
for f in os.listdir(backup_dir):
    os.remove(os.path.join(backup_dir, f))

# sample rate, Hz
SAMPLE_RATE = 16000
HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 10086  # Port to listen on (non-privileged ports are > 1023)

data_layer = AudioDataLayer(sample_rate=SAMPLE_RATE)
data_loader = DataLoader(data_layer, batch_size=1, collate_fn=data_layer.collate_fn)

STEP = 0.6
WINDOW_SIZE = 0.8
CHANNELS = 1 
RATE = 16000
FRAME_LEN = STEP
THRESHOLD = 0.5
CHUNK_SIZE = int(STEP * RATE)

asr = FrameASR(frame_len=FRAME_LEN, frame_overlap=1, offset=4)
asr.reset()
vad = FrameVAD(threshold=THRESHOLD, frame_len=FRAME_LEN, frame_overlap=1, offset=0)
vad.reset()

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.connect((HOST, PORT))
    s.send("hellohello".encode("utf-8"))
    text = []
    vad_text = [0,'','']
    empty_counter = 0
    while True:
        file_name = (s.recv(1024)).decode("utf-8")
        wf = wave.open(file_name, 'rb')
        data = wf.readframes(CHUNK_SIZE)
        out_text = ''
        while data != b'':
            signal = np.frombuffer(data, dtype=np.int16)
            vad_text, pdata = vad.transcribe(signal)
            if vad_text[0]:
                text = asr.transcribe(signal)
            if len(text):
                out_text += text
                empty_counter = asr.offset
            elif empty_counter > 0:
                empty_counter -= 1
                if empty_counter == 0:
                    print('\n',end='')
                    asr.reset()
                    vad.reset()
            data = wf.readframes(CHUNK_SIZE)
        msg = ("{}|{}".format(file_name, out_text)).encode("utf-8")
        s.send(msg)
