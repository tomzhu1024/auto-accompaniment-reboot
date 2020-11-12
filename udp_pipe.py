import json
import socket

PORT = 34001


class UDPReceiver:
    def __init__(self):
        self._s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._s.bind(('127.0.0.1', PORT))
        self._s.setblocking(0)

    def __call__(self):
        try:
            data, _ = self._s.recvfrom(65535)
        except socket.error:
            return
        else:
            return json.loads(data.decode('utf-8'))

    def close(self):
        self._s.close()


class UDPSender:
    def __init__(self):
        self._s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def __call__(self, data):
        self._s.sendto(json.dumps(data).encode('utf-8'), ('127.0.0.1', PORT))

    def close(self):
        self._s.close()
