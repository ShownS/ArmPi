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
# Set detection color
def setTargetColor(target_color):
    global __target_color

    #print("COLOR", target_color)
    __target_color = target_color
    return (True, ())

# Find the outline with the largest area
# The parameter is a list of contours to be compared
def getAreaMaxContour(contours):
    contour_area_temp = 0
    contour_area_max = 0
    area_max_contour = None

    for c in contours:  # Traversing all contours
        contour_area_temp = math.fabs(cv2.contourArea(c))  # Calculate contour area
        if contour_area_temp > contour_area_max:
            contour_area_max = contour_area_temp
            if contour_area_temp > 300:  # Only set as largest area contour if the area is greater than 300, filtering out interference
                area_max_contour = c

    return area_max_contour, contour_area_max  # Returns largest contour

# The angle where the gripper closes
servo1 = 500

# Initial position
def initMove():
    Board.setBusServoPulse(1, servo1 - 50, 300)
    Board.setBusServoPulse(2, 500, 500)
    AK.setPitchRangeMoving((0, 10, 10), -30, -30, -90, 1500)

def setBuzzer(timer):
    Board.setBuzzer(0)
    Board.setBuzzer(1)
    time.sleep(timer)
    Board.setBuzzer(0)

# Set expansion board LED to the color being tracked
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

# Use threading.Event objects to control state
_running = threading.Event()
_running.set()
_stop = threading.Event()
_stop.clear()

# Initialize arm position
def init():
    initMove()

# Start tracking
def start():
    global _stop
    _stop.clear()
    _running.set()

# Stop tracking
def stop():
    global _stop
    _stop.set()
    _running.clear()
    initMove()

class Motion:
    def __init__(
        self,
        gripper_id: int = 1,
        wrist_id: int = 2,
        grip_closed: int = 500,
        grip_open_delta: int = 280,
        grip_release_delta: int = 200,
        grip_relax_delta: int = 50,
        wrist_center: int = 500,
        home_xyz: Tuple[float, float, float] = (0, 10, 10),
        home_rpy: Tuple[float, float, float] = (-30, -30, -90),
        home_time_ms: int = 1500,
        y_offset: float = 2.0,
        track_z: float = 5.0,
        track_time_ms: int = 20,
        track_sleep_s: float = 0.02,
        approach_z: float = 7.0,
        pick_z: float = 2.0,
        pick_z_sort: float = 1.5,
        lift_z: float = 12.0,
        place_hover_dz: float = 3.0,
        place_hover_ms: int = 500,
        place_ms: int = 1000,
        stack_dz: float = 2.5,
        stack_levels: int = 3,
        stack_pause_s: float = 3.0,
        sort_bins: Optional[Dict[str, Tuple[float, float, float]]] = None,
        stack_base: Optional[Dict[str, Tuple[float, float, float]]] = None,
    ):
        self.AK = ArmIK()
        self._lock = threading.Lock()

        self.gripper_id = int(gripper_id)
        self.wrist_id = int(wrist_id)

        self.grip_closed = int(grip_closed)
        self.grip_open_delta = int(grip_open_delta)
        self.grip_release_delta = int(grip_release_delta)
        self.grip_relax_delta = int(grip_relax_delta)

        self.wrist_center = int(wrist_center)

        self.home_xyz = tuple(home_xyz)
        self.home_rpy = tuple(home_rpy)
        self.home_time_ms = int(home_time_ms)

        self.y_offset = float(y_offset)
        self.track_z = float(track_z)
        self.track_time_ms = int(track_time_ms)
        self.track_sleep_s = float(track_sleep_s)

        self.approach_z = float(approach_z)
        self.pick_z = float(pick_z)
        self.pick_z_sort = float(pick_z_sort)
        self.lift_z = float(lift_z)

        self.place_hover_dz = float(place_hover_dz)
        self.place_hover_ms = int(place_hover_ms)
        self.place_ms = int(place_ms)

        self.stack_dz = float(stack_dz)
        self.stack_levels = max(1, int(stack_levels))
        self.stack_pause_s = float(stack_pause_s)

        if sort_bins is None:
            self.sort_bins = {
                "red":   (-15 + 0.5, 12 - 0.5, 1.5),
                "green": (-15 + 0.5,  6 - 0.5, 1.5),
                "blue":  (-15 + 0.5,  0 - 0.5, 1.5),
            }
        else:
            self.sort_bins = dict(sort_bins)

        if stack_base is None:
            self.stack_base = {
                "red":   (-15 + 1, -7 - 0.5, 1.5),
                "green": (-15 + 1, -7 - 0.5, 1.5),
                "blue":  (-15 + 1, -7 - 0.5, 1.5),
            }
        else:
            self.stack_base = dict(stack_base)

        self._stack_level = {c: 0 for c in ("red", "green", "blue")}
        self.move_square = False

    def _sleep_ms(self, ms: int):
        time.sleep(max(0.0, ms) / 1000.0)

    def _move_auto(self, xyz, rpy) -> bool:
        result = self.AK.setPitchRangeMoving(xyz, rpy[0], rpy[1], rpy[2])
        if result is False:
            return False
        self._sleep_ms(int(result[2]))
        return True

    def _move(self, xyz, rpy, ms: int) -> bool:
        self.AK.setPitchRangeMoving(xyz, rpy[0], rpy[1], rpy[2], int(ms))
        self._sleep_ms(ms)
        return True

    def buzz(self, s: float = 0.1):
        Board.setBuzzer(0)
        Board.setBuzzer(1)
        time.sleep(max(0.0, s))
        Board.setBuzzer(0)

    def open(self, ms: int = 500):
        pulse = self.grip_closed - self.grip_open_delta
        Board.setBusServoPulse(self.gripper_id, int(pulse), int(ms))
        self._sleep_ms(ms)

    def close(self, ms: int = 500):
        Board.setBusServoPulse(self.gripper_id, int(self.grip_closed), int(ms))
        self._sleep_ms(ms)

    def release(self, ms: int = 500):
        pulse = self.grip_closed - self.grip_release_delta
        Board.setBusServoPulse(self.gripper_id, int(pulse), int(ms))
        self._sleep_ms(ms)

    def wrist0(self, ms: int = 500):
        Board.setBusServoPulse(self.wrist_id, int(self.wrist_center), int(ms))
        self._sleep_ms(ms)

    def wrist(self, x: float, y: float, angle_deg: float, ms: int = 500) -> int:
        p = int(getAngle(x, y, angle_deg))
        Board.setBusServoPulse(self.wrist_id, p, int(ms))
        self._sleep_ms(ms)
        return p

    def home(self):
        with self._lock:
            relax = self.grip_closed - self.grip_relax_delta
            Board.setBusServoPulse(self.gripper_id, int(relax), 300)
            time.sleep(0.5)
            self.wrist0(500)
            self._move(self.home_xyz, self.home_rpy, self.home_time_ms)

    def stop(self):
        self.home()

    def first(self, X: float, Y: float) -> bool:
        with self._lock:
            return self._move_auto((X, Y - self.y_offset, self.track_z), (-90, -90, 0))

    def track(self, x: float, y: float) -> bool:
        with self._lock:
            self._move((x, y - self.y_offset, self.track_z), (-90, -90, 0), self.track_time_ms)
        time.sleep(self.track_sleep_s)
        return True

    def approach(self, X: float, Y: float) -> bool:
        with self._lock:
            return self._move_auto((X, Y, self.approach_z), (-90, -90, 0))

    def pick(self, X: float, Y: float, rot: float, z: Optional[float] = None) -> bool:
        if z is None:
            z = self.pick_z
        with self._lock:
            self.open(500)
            self.wrist(X, Y, rot, 500)

            self._move((X, Y, float(z)), (-90, -90, 0), 1000)
            time.sleep(0.5)

            self.close(500)
            time.sleep(0.3)

            self.wrist0(500)
            self._move((X, Y, self.lift_z), (-90, -90, 0), 1000)
            time.sleep(0.2)
        return True

    def place(self, x: float, y: float, z: float) -> bool:
        with self._lock:
            ok = self._move_auto((x, y, self.lift_z), (-90, -90, 0))
            if not ok:
                return False

            self.wrist(x, y, -90, 500)

            self._move((x, y, float(z) + self.place_hover_dz), (-90, -90, 0), self.place_hover_ms)
            self._move((x, y, float(z)), (-90, -90, 0), self.place_ms)

            self.release(500)
            time.sleep(0.2)

            self._move((x, y, self.lift_z), (-90, -90, 0), 800)
            time.sleep(0.2)
        return True

    def pick_place(self, X: float, Y: float, rot: float, dest: Tuple[float, float, float], *, z: Optional[float] = None) -> bool:
        if not self.approach(X, Y):
            return False
        if not self.pick(X, Y, rot, z=z):
            return False
        if not self.place(*dest):
            return False
        self.home()
        return True

    def sort(self, X: float, Y: float, rot: float, color: str, *, use_sort_height: bool = False) -> bool:
        if color not in self.sort_bins:
            return False

        self.buzz(0.1)

        if not self.approach(X, Y):
            return False

        z = self.pick_z_sort if use_sort_height else self.pick_z
        if not self.pick(X, Y, rot, z=z):
            return False

        if not self.place(*self.sort_bins[color]):
            return False

        self.home()
        return True

    def _next_stack_z(self, color: str) -> float:
        _, _, base_z = self.stack_base[color]
        level = self._stack_level[color]
        z = base_z + level * self.stack_dz

        self._stack_level[color] = (level + 1) % self.stack_levels

        if abs(z - base_z) < 1e-9:
            self.move_square = True
            time.sleep(self.stack_pause_s)
            self.move_square = False

        return z

    def stack(self, X: float, Y: float, rot: float, color: str) -> bool:
        if color not in self.stack_base:
            return False

        self.buzz(0.1)

        bx, by, _ = self.stack_base[color]
        bz = self._next_stack_z(color)

        if not self.approach(X, Y):
            return False
        if not self.pick(X, Y, rot, z=self.pick_z):
            return False
        if not self.place(bx, by, bz):
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

    def _reset_stability(self):
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

            #cv2.circle(frame_resize,(int(center_x), int(center_y)),5,range_rgb[max_color],-1)

            cv2.putText(frame_resize,max_color,(int(center_x) + 10, int(center_y)),cv2.FONT_HERSHEY_SIMPLEX,
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
            self._reset_stability()

        return frame_resize, detection


if __name__ == '__main__':
    init()
    start()

    perception = Perception(target_colors=('red','green','blue'))

    my_camera = Camera.Camera()
    my_camera.camera_open()

    # Create Motion ONCE (not inside the loop)
    # motion = Motion()
    # motion.home()

    # # Ask user once which operation to run
    # mode_in = input("Select mode: [p]ick_place, [s]ort, s[t]ack : ").strip().lower()
    # if mode_in in ("p", "pick", "pick_place", "pickplace"):
    #     mode = "pick_place"
    # elif mode_in in ("t", "stack", "pallet", "palletize", "palletizing"):
    #     mode = "stack"
    # else:
    #     mode = "sort"
    # print(f"Running mode: {mode}")

    # Prevent repeated triggering while ready stays True
    busy = False

    ready_prev = False

    try:
        while True:
            img = my_camera.frame
            if img is None:
                time.sleep(0.01)
                continue

            annotated, detection = perception.process(img)

            ready_now = bool(detection and detection.get("ready", False))

            if (not busy) and detection and (not ready_now):
                wx = detection.get("world_x")
                wy = detection.get("world_y")
                # if (wx is not None) and (wy is not None):
                    # motion.track(wx, wy)

            if (not busy) and ready_now and (not ready_prev):
                busy = True

                X = detection["world_X_avg"]
                Y = detection["world_Y_avg"]
                rot = detection["rotation_angle"]
                color = detection["confirmed_color"]

                # if mode == "sort":
                #     motion.sort(X, Y, rot, color)


                # elif mode == "stack":
                #     motion.stack(X, Y, rot, color)

                # else:  # pick_place
                #     # choose a simple fixed destination (you can change this)
                #     dest = (-15 + 0.5, 12 - 0.5, 1.5)
                #     motion.pick_place(X, Y, rot, dest)
                # after motion finishes
                perception._reset_stability()
                ready_prev = False
                busy = False

            # Optional: print stable detection results
            if detection and detection.get("stable", False):
                print("Stable:", detection)

            ready_prev = ready_now

            cv2.imshow('Perception Demo', annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or q
                break
    finally:
        my_camera.camera_close()
        cv2.destroyAllWindows()
        stop()
        exit()