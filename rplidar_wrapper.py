import logging
import logcolor
import threading
from time import sleep
from datetime import datetime, timedelta

import numpy as np
from rplidar import RPLidar, RPLidarException

class LidarWrapper:

    def __init__(self):
        self._scans = dict()
        self.timestamps = dict()
        self.running = True
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.thread = threading.Thread(target=self.start)
        self.thread.do_run = True
        self.thread.start()


    def start(self):
        while threading.current_thread().do_run:
            try:
                rpl = RPLidar('/dev/ttyUSB1')
                for i, scan in enumerate(rpl.iter_measurments()):
                    _, _, angle, distance = scan
                    # lidar is upside down, so change it here
                    angle = int(angle)
                    distance /= 1000.0
                    if 0.0 < distance < 0.45 and (angle < 50 or 360 - 40 < angle):
                        if angle > 90:
                            angle -= 360
                        self._scans[angle] = distance
                        self.timestamps[angle] = datetime.now()
            except RPLidarException:
                continue
            break

    @property
    def scans(self):
        res = {}
        for angle, dist in self._scans.items():
             if self.timestamps.get(angle, datetime.now() - timedelta(seconds=10)) > datetime.now() - timedelta(milliseconds=200):
                res[angle] = dist
        return res

    def stop(self):
        self.logger.info('Stopping scanning')
        self.thread.do_run = False
        self.thread.join()
        

if __name__ == '__main__':
    l = LidarWrapper()
    sleep(3.0)
    l.stop()
    print(l.scans)
