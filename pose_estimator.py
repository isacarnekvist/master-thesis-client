from time import sleep
from pprint import pprint

import numpy as np
from numpy.linalg import norm
from numpy import sin, cos, arccos, dot
from skimage.transform import hough_line, hough_line_peaks

from rplidar_wrapper import LidarWrapper


def cube_pose(points):
    h, w = 150, 200
    img = np.zeros((h, w), dtype=np.uint64)
    for x, y in points:
        i, j = int(x * 500), int((y + 0.20) * 500)
        img[min(j, h - 1), min(i, w - 1)] = 1

    hspace, angles, dists = hough_line(img, theta=np.linspace(-np.pi / 2, np.pi / 2, 360))
    hspace, angles, dists = hough_line_peaks(hspace, angles, dists)
    max_i = np.argmax(hspace)
    theta = angles[max_i]
    d = np.array([np.cos(theta), np.sin(theta)]) * dists[max_i]
    v = np.array([np.cos(theta + np.pi / 2), np.sin(theta + np.pi / 2)])

    segment = []
    for x, y in points:
        i, j = int(x * 500), int((y + 0.20) * 500)
        p = np.array([i, j])
        gamma = np.arccos(np.dot(p - d, v) / np.linalg.norm(p - d))
        s = np.sin(np.arccos(np.dot((p - d), v) / np.linalg.norm(p - d))) * np.linalg.norm(p - d)
        if s < 2:
            segment.append((x, y))

    v1 = np.array(segment[0])
    v2 = np.array(segment[-1])
    return v1 + (v2 - v1) / 2 + 0.02 * np.array([np.cos(theta), np.sin(theta)]), theta


lw = LidarWrapper()
try:
    while True:
        sleep(0.2)
        points = [
            (dist * np.cos(np.deg2rad(angle)), dist * np.sin(np.deg2rad(angle)))
            for angle, dist in sorted(lw.scans.items(), key=lambda x: x[0])
        ]
        if points:
            pprint(cube_pose(points))
except KeyboardInterrupt:
    lw.stop()
    exit(0)
