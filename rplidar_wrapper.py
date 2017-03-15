import logging
import logcolor
import threading
from time import sleep
from datetime import datetime, timedelta

import numpy as np
from pose_estimator import cube_pose
from rplidar import RPLidar, RPLidarException

class LidarWrapper:

    def __init__(self):
        self._scans = dict()
        self.cube_pose = np.array([0.0, 0.0])
        self.timestamps = dict()
        self.running = True
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.thread = threading.Thread(target=self.start)
        self.thread.do_run = True
        self.thread.start()


    def start(self):
        scans = dict()
        while threading.current_thread().do_run:
            try:
                rpl = RPLidar('/dev/ttyUSB1')
                for i, scan in enumerate(rpl.iter_measurments()):
                    if not threading.current_thread().do_run:
                        break
                    new_scan, quality, angle, distance = scan
                    if new_scan:
                        if len(scans) > 0:
                            x, y, _ = cube_pose(scan_dict=scans)
                            self.cube_pose = 0.7 * self.cube_pose + 0.3 * np.array([x, y])
                        self._scans = scans
                        scans = dict()
                    # lidar is upside down, so change it here
                    distance /= 1000.0
                    if 0.0 < distance < 0.40 and (angle < 20.0 or angle > 360.0 - 20.0):
                        scans[angle] = distance
            except RPLidarException:
                continue
            break

    @property
    def scans(self):
        return self._scans

    @property
    def cube_visible(self):
        return len(self.scans) > 0

    def stop(self):
        self.logger.info('Stopping scanning')
        self.thread.do_run = False
        self.thread.join()
        

if __name__ == '__main__':
    l = LidarWrapper()
    while True:
        sleep(0.2)
        print(l.cube_pose)
    l.stop()
