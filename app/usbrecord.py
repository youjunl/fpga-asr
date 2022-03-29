from logging import raiseExceptions
import pyaudio
import math
import struct
import wave
import time
import os
from . import server
import audioop
Threshold = 10

SHORT_NORMALIZE = (1.0/32768.0)
chunk = 1600
CHANNELS = 1
RATE = 16000
swidth = 2
FORMAT = pyaudio.paInt16
TIMEOUT_LENGTH = 1

f_name_directory = r'./audio'
is_not_the_first_one_flag = 0

class Recorder:
    
    @staticmethod
    def rms(frame):
        count = len(frame) / swidth
        format = "%dh" % (count)
        shorts = struct.unpack(format, frame)
        sum_squares = 0.0
        for sample in shorts:
            n = sample * SHORT_NORMALIZE
            sum_squares += n * n
        rms = math.pow(sum_squares / count, 0.5)
        return rms * 1000

    def resample(self, data, rate):
        (newfragment, state) = audioop.ratecv(data, 2, 1, rate, 16000, None)
        return newfragment

    def __init__(self, chunk_size=chunk):
        print("CHUNK: %d"%chunk_size)
        self.recording_symbol = 0
        self.p = pyaudio.PyAudio()
        print('Available audio input devices:')
        input_devices = []
        input_devices_name = []
        for i in range(self.p.get_device_count()):
            dev = self.p.get_device_info_by_index(i)
            if dev.get('maxInputChannels'):
                input_devices.append(i)
                dev_name = dev.get('name')
                input_devices_name.append(dev_name)
                print(i, dev_name)
        if len(input_devices):
            dev_idx = -2
            self.stream = self.p.open(format=FORMAT,
                                    channels=CHANNELS,
                                    rate=RATE,
                                    input=True,
                                    output=True,
                                    frames_per_buffer=chunk_size,
                                    input_device_index=dev_idx,
            )
        else:
            raiseExceptions("ERROR: No audio input device found.")

    def record(self):
        print('Sound detected, recording begin')
        rec = []
        current = time.time()
        end = time.time() + TIMEOUT_LENGTH
        while current <= end:
            data = self.stream.read(chunk)
            if self.rms(data) >= Threshold: end = time.time() + TIMEOUT_LENGTH
            current = time.time()
            rec.append(data)
        self.write(b''.join(rec))

    def write(self, recording):
        global is_not_the_first_one_flag
        n_files = len(os.listdir(f_name_directory))
        filename = os.path.join(f_name_directory, '{}.wav'.format(n_files))
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(FORMAT))
        wf.setframerate(16000)
        wf.writeframes(recording)
        wf.close()
        print('Written to file: {}'.format(filename))
        if(is_not_the_first_one_flag ==0):
            server.audio_client_socket.send(filename.encode("utf-8"))
            is_not_the_first_one_flag = 1
        else:
            server.server_send_queue.put(filename)       
        print('Returning to listening')

    def listen(self):
        print('Listening beginning')
        while True:
            if(self.recording_symbol == 1):
                #time.sleep(0.01)
                input = self.stream.read(chunk, exception_on_overflow = False)
                rms_val = self.rms(input)
                if rms_val > Threshold:
                    self.record()

            else:
                print("Stop recording")
                return

if __name__ == "__main__":
    a = Recorder()
    a.listen()