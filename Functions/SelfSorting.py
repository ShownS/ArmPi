#!/usr/bin/python3
# coding=utf8
import sys
sys.path.append('/home/pi/ArmPi/')
import cv2
import time
import Camera
import threading
from LABConfig import *
from ArmIK.Transform import *
from ArmIK.ArmMoveIK import *
import HiwonderSDK.Board as Board
from CameraCalibration.CalibrationConfig import *

from typing import Dict, Tuple, Optional

import math
import numpy as np

if sys.version_info.major == 2:
    print('Please run this program with python3!')
    sys.exit(0)

AK = ArmIK()

range_rgb = {
    'red': (0, 0, 255),
    'blue': (255, 0, 0),
    'green': (0, 255, 0),
    'black': (0, 0, 0),
    'white': (255, 255, 255),
}

__target_color = ('red',)
def setTargetColor(target_color):
    global __target_color
    __target_color = target_color
    return (True, ())

def getAreaMaxContour(contours):
    contour_area_temp = 0
    contour_area_max = 0
    area_max_contour = None

    for c in contours:
        contour_area_temp = math.fabs(cv2.contourArea(c))
        if contour_area_temp > contour_area_max:
            contour_area_max = contour_area_temp
            if contour_area_temp > 300:
                area_max_contour = c

    return area_max_contour, contour_area_max

servo1 = 500

def initMove():
    Board.setBusServoPulse(1, servo1 - 50, 300)
    Board.setBusServoPulse(2, 500, 500)
    AK.setPitchRangeMoving((0, 10, 10), -30, -30, -90, 1500)

def setBuzzer(timer):
    Board.setBuzzer(0)
    Board.setBuzzer(1)
    time.sleep(timer)
    Board.setBuzzer(0)

def set_rgb(color):
    if color == "red":
        Board.RGB.setPixelColor(0, Board.PixelColor(255, 0, 0))
        Board.RGB.setPixelColor(1, Board.PixelColor(255, 0, 0))
        Board.RGB.show()
    elif color == "green":
        Board.RGB.setPixelColor(0, Board.PixelColor(0, 255, 0))
        Board.RGB.setPixelColor(1, Board.PixelColor(0, 255, 0))
        Board.RGB.show()
    elif color == "blue":
        Board.RGB.setPixelColor(0, Board.PixelColor(0, 0, 255))
        Board.RGB.setPixelColor(1, Board.PixelColor(0, 0, 255))
        Board.RGB.show()
    else:
        Board.RGB.setPixelColor(0, Board.PixelColor(0, 0, 0))
        Board.RGB.setPixelColor(1, Board.PixelColor(0, 0, 0))
        Board.RGB.show()

_running = threading.Event()
_running.set()
_stop = threading.Event()
_stop.clear()

def init():
    initMove()

def start():
    global _stop
    _stop.clear()
    _running.set()

def stop():
    global _stop
    _stop.set()
    _running.clear()
    initMove()

class Motion:
    def __init__(self):
        self.AK = ArmIK()

        self.gripper_id = 1
        self.wrist_id = 2

        self.grip_closed = 500
        self.grip_open = self.grip_closed - 280
        self.grip_release = self.grip_closed - 200
        self.grip_relax = self.grip_closed - 50

        self.wrist_center = 500

        self.home_xyz = (0, 10, 10)
        self.home_rpy = (-30, -30, -90)

        self.approach_z = 7.0
        self.pick_z = 2.0
        self.lift_z = 12.0

        self.sort_bins = {
            "red":   (-14.5, 11.5, 1.5),
            "green": (-14.5, 5.5, 1.5),
            "blue":  (-14.5, -0.5, 1.5),
        }

    def home(self):
        Board.setBusServoPulse(self.gripper_id, int(self.grip_relax), 300)
        time.sleep(0.5)
        Board.setBusServoPulse(self.wrist_id, int(self.wrist_center), 500)
        time.sleep(0.5)
        self.AK.setPitchRangeMoving(self.home_xyz, self.home_rpy[0], self.home_rpy[1], self.home_rpy[2], 1500)
        time.sleep(1.5)

    def pick(self, X, Y, rot):
        result = self.AK.setPitchRangeMoving((X, Y, self.approach_z), -90, -90, 0)
        if result is False:
            return False
        time.sleep(result[2] / 1000.0)

        Board.setBusServoPulse(self.gripper_id, int(self.grip_open), 500)
        time.sleep(0.5)

        wrist_angle = getAngle(X, Y, rot)
        Board.setBusServoPulse(self.wrist_id, int(wrist_angle), 500)
        time.sleep(0.5)

        self.AK.setPitchRangeMoving((X, Y, self.pick_z), -90, -90, 0, 1000)
        time.sleep(1.0)

        Board.setBusServoPulse(self.gripper_id, int(self.grip_closed), 500)
        time.sleep(0.5)

        Board.setBusServoPulse(self.wrist_id, int(self.wrist_center), 500)
        time.sleep(0.5)

        self.AK.setPitchRangeMoving((X, Y, self.lift_z), -90, -90, 0, 1000)
        time.sleep(1.0)

        return True

    def place(self, x, y, z):
        result = self.AK.setPitchRangeMoving((x, y, self.lift_z), -90, -90, 0)
        if result is False:
            return False
        time.sleep(result[2] / 1000.0)

        wrist_angle = getAngle(x, y, -90)
        Board.setBusServoPulse(self.wrist_id, int(wrist_angle), 500)
        time.sleep(0.5)

        self.AK.setPitchRangeMoving((x, y, z + 3), -90, -90, 0, 500)
        time.sleep(0.5)

        self.AK.setPitchRangeMoving((x, y, z), -90, -90, 0, 1000)
        time.sleep(1.0)

        Board.setBusServoPulse(self.gripper_id, int(self.grip_release), 500)
        time.sleep(0.5)

        self.AK.setPitchRangeMoving((x, y, self.lift_z), -90, -90, 0, 800)
        time.sleep(0.8)

        return True

    def sort(self, X, Y, rot, color):
        if color not in self.sort_bins:
            return False

        Board.setBuzzer(0)
        Board.setBuzzer(1)
        time.sleep(0.1)
        Board.setBuzzer(0)

        if not self.pick(X, Y, rot):
            return False

        x, y, z = self.sort_bins[color]
        if not self.place(x, y, z):
            return False

        self.home()
        return True

class Perception:
    def __init__(self, target_colors=('red', 'green', 'blue')):
        self.target_colors = target_colors
        self.camera_angle = 0.0

        self.world_x = 0.0
        self.world_y = 0.0
        self.world_X = 0.0
        self.world_Y = 0.0

        self.rotation_angle = 0.0
        self.detect_color = 'None'

        self.color_list = []
        self.coordinate_list = []

        self.stable_count = 0
        self.confirmed_color = None
        self.world_X_avg = None
        self.world_Y_avg = None

    def reset(self):
        self.color_list = []
        self.coordinate_list = []
        self.stable_count = 0
        self.confirmed_color = None
        self.world_X_avg = None
        self.world_Y_avg = None

    def process(self, img):
        if img is None:
            return None, None

        frame = img.copy()
        frame_resize = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_NEAREST)

        frame_gb = cv2.GaussianBlur(frame_resize, (3, 3), 3)
        frame_lab = cv2.cvtColor(frame_gb, cv2.COLOR_BGR2LAB)

        color_area_max = 0
        area_max_contour = None
        max_color = None

        for color in self.target_colors:
            if color not in color_range:
                continue

            frame_mask = cv2.inRange(frame_lab, color_range[color][0], color_range[color][1])
            opened = cv2.morphologyEx(frame_mask, cv2.MORPH_OPEN, np.ones((6, 6), np.uint8))
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((6, 6), np.uint8))
            contours = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]

            areaMaxContour, area_max = getAreaMaxContour(contours)
            if areaMaxContour is not None and area_max > color_area_max:
                color_area_max = area_max
                area_max_contour = areaMaxContour
                max_color = color

        detection = None

        if area_max_contour is not None and max_color is not None:
            rect = cv2.minAreaRect(area_max_contour)
            box = np.int0(cv2.boxPoints(rect))
            cv2.drawContours(frame_resize, [box], -1, range_rgb[max_color], 2)
            
            self.rotation_angle = rect[2]
            self.detect_color = max_color

            center_x, center_y = rect[0]
            self.world_x, self.world_y = convertCoordinate(center_x, center_y, size=(640, 480))
            self.world_X, self.world_Y = self.world_x, self.world_y

            self.color_list.append(self.detect_color)
            self.coordinate_list.append((self.world_X, self.world_Y))

            cv2.putText(frame_resize,f"{max_color}, ({self.world_X}, {self.world_Y})",(int(center_x) + 10, int(center_y)),cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,range_rgb[max_color],2)

            if len(self.coordinate_list) >= 6:
                self.stable_count += 1

                if self.stable_count >= 5:
                    from collections import Counter
                    most_common = Counter(self.color_list).most_common(1)[0][0]
                    self.confirmed_color = most_common

                    xs = [p[0] for p in self.coordinate_list]
                    ys = [p[1] for p in self.coordinate_list]
                    self.world_X_avg = float(sum(xs) / len(xs))
                    self.world_Y_avg = float(sum(ys) / len(ys))

                    detection = {
                        "stable": True,
                        "ready": True,
                        "confirmed_color": self.confirmed_color,
                        "world_X_avg": self.world_X_avg,
                        "world_Y_avg": self.world_Y_avg,
                        "rotation_angle": float(self.rotation_angle),
                        "world_x": float(self.world_x),
                        "world_y": float(self.world_y),
                    }
                else:
                    detection = {
                        "stable": True,
                        "ready": False,
                        "confirmed_color": None,
                        "world_X_avg": None,
                        "world_Y_avg": None,
                        "rotation_angle": float(self.rotation_angle),
                        "world_x": float(self.world_x),
                        "world_y": float(self.world_y),
                    }
            else:
                detection = {
                    "stable": False,
                    "ready": False,
                    "confirmed_color": None,
                    "world_X_avg": None,
                    "world_Y_avg": None,
                    "rotation_angle": float(self.rotation_angle),
                    "world_x": float(self.world_x),
                    "world_y": float(self.world_y),
                }
        else:
            self.reset()

        return frame_resize, detection


if __name__ == '__main__':
    init()
    start()

    perception = Perception(target_colors=('red', 'green', 'blue'))

    my_camera = Camera.Camera()
    my_camera.camera_open()

    motion = Motion()
    motion.home()

    busy = False

    try:
        while True:
            img = my_camera.frame
            if img is None:
                time.sleep(0.01)
                continue

            annotated, detection = perception.process(img)

            if detection and detection.get("ready", False) and not busy:
                busy = True

                X = detection["world_X_avg"]
                Y = detection["world_Y_avg"]
                rot = detection["rotation_angle"]
                color = detection["confirmed_color"]

                motion.sort(X, Y, rot, color)

                perception.reset()
                busy = False

            cv2.imshow('Perception Demo', annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    finally:
        my_camera.camera_close()
        cv2.destroyAllWindows()
        stop()
        exit()