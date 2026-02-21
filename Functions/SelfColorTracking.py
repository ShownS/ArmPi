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

count = 0
track = False
_stop = False
get_roi = False
center_list = []
first_move = True
__isRunning = False
detect_color = 'None'
action_finish = True
start_pick_up = False
start_count_t1 = True
# Reset variables
def reset():
    global count
    global track
    global _stop
    global get_roi
    global first_move
    global center_list
    global __isRunning
    global detect_color
    global action_finish
    global start_pick_up
    global __target_color
    global start_count_t1
    
    count = 0
    _stop = False
    track = False
    get_roi = False
    center_list = []
    first_move = True
    __target_color = ()
    detect_color = 'None'
    action_finish = True
    start_pick_up = False
    start_count_t1 = True

# App initialization call
def init():
    print("ColorTracking Init")
    initMove()

# App start gameplay call
def start():
    global __isRunning
    reset()
    __isRunning = True
    print("ColorTracking Start")

# App stop gameplay
def stop():
    global _stop 
    global __isRunning
    _stop = True
    __isRunning = False
    print("ColorTracking Stop")

# App exit gameplay call
def exit():
    global _stop
    global __isRunning
    _stop = True
    __isRunning = False
    print("ColorTracking Exit")

rect = None
size = (640, 480)
rotation_angle = 0
unreachable = False
world_X, world_Y = 0, 0
world_x, world_y = 0, 0
# Arm movement thread
def move():
    global rect
    global track
    global _stop
    global get_roi
    global unreachable
    global __isRunning
    global detect_color
    global action_finish
    global rotation_angle
    global world_X, world_Y
    global world_x, world_y
    global center_list, count
    global start_pick_up, first_move

    # Place blocks of different colors at coordinates (x,y,z)
    coordinate = {
        'red':   (-15 + 0.5, 12 - 0.5, 1.5),
        'green': (-15 + 0.5, 6 - 0.5,  1.5),
        'blue':  (-15 + 0.5, 0 - 0.5,  1.5),
    }
    while True:
        if __isRunning:
            if first_move and start_pick_up: # When an object is first detected               
                action_finish = False
                set_rgb(detect_color)
                setBuzzer(0.1)               
                result = AK.setPitchRangeMoving((world_X, world_Y - 2, 5), -90, -90, 0) # If the runtime parameter is not filled in, the runtime will be adaptive.

                if result == False:
                    unreachable = True
                else:
                    unreachable = False
                time.sleep(result[2]/1000) # Time is the third parameter returned
                start_pick_up = False
                first_move = False
                action_finish = True
            elif not first_move and not unreachable: # This is not the first time an object has been detected
                set_rgb(detect_color)
                if track: # If it is the tracking stage
                    if not __isRunning: # Stop and exit flag detection
                        continue
                    AK.setPitchRangeMoving((world_x, world_y - 2, 5), -90, -90, 0, 20)
                    time.sleep(0.02)                    
                    track = False
                if start_pick_up: # if the object has not moved for a period of time, start clamping
                    action_finish = False
                    if not __isRunning: # Stop and exit flag detection
                        continue
                    Board.setBusServoPulse(1, servo1 - 280, 500)  # Claws open
                    # Calculate the angle the gripper needs to rotate
                    servo2_angle = getAngle(world_X, world_Y, rotation_angle)
                    Board.setBusServoPulse(2, servo2_angle, 500)
                    time.sleep(0.8)
                    
                    if not __isRunning:
                        continue
                    AK.setPitchRangeMoving((world_X, world_Y, 2), -90, -90, 0, 1000)  # Lower height
                    time.sleep(2)
                    
                    if not __isRunning:
                        continue
                    Board.setBusServoPulse(1, servo1, 500)  # Clamp closing
                    time.sleep(1)
                    
                    if not __isRunning:
                        continue
                    Board.setBusServoPulse(2, 500, 500)
                    AK.setPitchRangeMoving((world_X, world_Y, 12), -90, -90, 0, 1000)  # Robotic arm raised
                    time.sleep(1)
                    
                    if not __isRunning:
                        continue
                    # Sort and place the different colored blocks
                    result = AK.setPitchRangeMoving((coordinate[detect_color][0], coordinate[detect_color][1], 12), -90, -90, 0)   
                    time.sleep(result[2]/1000)
                    
                    if not __isRunning:
                        continue
                    servo2_angle = getAngle(coordinate[detect_color][0], coordinate[detect_color][1], -90)
                    Board.setBusServoPulse(2, servo2_angle, 500)
                    time.sleep(0.5)

                    if not __isRunning:
                        continue
                    AK.setPitchRangeMoving((coordinate[detect_color][0], coordinate[detect_color][1], coordinate[detect_color][2] + 3), -90, -90, 0, 500)
                    time.sleep(0.5)
                    
                    if not __isRunning:
                        continue
                    AK.setPitchRangeMoving((coordinate[detect_color]), -90, -90, 0, 1000)
                    time.sleep(0.8)
                    
                    if not __isRunning:
                        continue
                    Board.setBusServoPulse(1, servo1 - 200, 500)  # Open claws, release block
                    time.sleep(0.8)
                    
                    if not __isRunning:
                        continue                    
                    AK.setPitchRangeMoving((coordinate[detect_color][0], coordinate[detect_color][1], 12), -90, -90, 0, 800)
                    time.sleep(0.8)

                    initMove()  # Return to initial position
                    time.sleep(1.5)

                    detect_color = 'None'
                    first_move = True
                    get_roi = False
                    action_finish = True
                    start_pick_up = False
                    set_rgb(detect_color)
                else:
                    time.sleep(0.01)
        else:
            if _stop:
                _stop = False
                Board.setBusServoPulse(1, servo1 - 70, 300)
                time.sleep(0.5)
                Board.setBusServoPulse(2, 500, 500)
                AK.setPitchRangeMoving((0, 10, 10), -30, -30, -90, 1500)
                time.sleep(1.5)
            time.sleep(0.01)

# Run child thread
# th = threading.Thread(target=move)
# th.setDaemon(True)
# th.start()

t1 = 0
roi = ()
last_x, last_y = 0, 0



class Perception:
    def __init__(
        self,
        target_colors=("red", "green", "blue"),
        min_area=2500,
        vote_len=3,
        stable_dist=0.5,
        stable_time=1.0,
        require_color_confirm=True,
        ):
        self.target_colors = tuple(target_colors)
        self.min_area = float(min_area)

        # voting
        self.vote_len = int(vote_len)
        self.color_votes = []          # list[int] codes 1/2/3
        self.confirmed_color = "None"  # 'red'/'green'/'blue'/'None'
        self.draw_color = range_rgb.get("black", (0, 0, 0))

        # stability gate
        self.stable_dist = float(stable_dist)
        self.stable_time = float(stable_time)
        self.require_color_confirm = bool(require_color_confirm)

        self.last_x = None
        self.last_y = None
        self.center_list = []   # [x1,y1,x2,y2,...]
        self.count = 0
        self.t_stable_start = None

        # ROI optimization state (borrowed idea from the stock scripts)
        self.get_roi = False
        self.roi = None

        # final stable outputs
        self.world_x = None
        self.world_y = None
        self.world_X_avg = None
        self.world_Y_avg = None
        self.rotation_angle = None
        self.ready = False  # becomes True only when stable + (optionally) color confirmed

    def reset(self):
        self.color_votes = []
        self.confirmed_color = "None"
        self.draw_color = range_rgb.get("black", (0, 0, 0))

        self.last_x = None
        self.last_y = None
        self.center_list = []
        self.count = 0
        self.t_stable_start = None

        self.get_roi = False
        self.roi = None

        self.world_x = None
        self.world_y = None
        self.world_X_avg = None
        self.world_Y_avg = None
        self.rotation_angle = None
        self.ready = False

    def get_area_max_contour(self, contours):
        area_max = 0
        area_max_contour = None
        for c in contours:
            area = abs(cv2.contourArea(c))
            if area > area_max:
                area_max = area
                area_max_contour = c
        return area_max_contour, area_max

    def _color_to_code(self, color_name):
        if color_name == "red":
            return 1
        if color_name == "green":
            return 2
        if color_name == "blue":
            return 3
        return 0

    def _code_to_color(self, code):
        return {1: "red", 2: "green", 3: "blue"}.get(code, "None")

    def preprocess(self, img_bgr):
        img_resize = cv2.resize(img_bgr, size, interpolation=cv2.INTER_NEAREST)
        img_blur = cv2.GaussianBlur(img_resize, (11, 11), 11)

        # ROI optimization: if we have ROI from last time, restrict search area
        if self.get_roi and self.roi is not None:
            img_blur = getMaskROI(img_blur, self.roi, size)
            self.get_roi = False  # one-shot mask like the original scripts

        img_lab = cv2.cvtColor(img_blur, cv2.COLOR_BGR2LAB)
        return img_resize, img_lab

    def segment(self, img_lab, color_name):
        mask = cv2.inRange(img_lab, color_range[color_name][0], color_range[color_name][1])
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((6, 6), np.uint8))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((6, 6), np.uint8))
        return closed

    def find_best_candidate(self, img_lab):
        best_color = None
        best_contour = None
        best_area = 0

        for c in self.target_colors:
            if c not in color_range:
                continue
            closed = self.segment(img_lab, c)
            contours = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]

            contour, area = self.get_area_max_contour(contours)
            if contour is not None and area > best_area:
                best_area = area
                best_contour = contour
                best_color = c

        return best_color, best_contour, best_area

    def compute_pose(self, contour):
        rect = cv2.minAreaRect(contour)
        box = np.int0(cv2.boxPoints(rect))
        roi = getROI(box)
        img_centerx, img_centery = getCenter(rect, roi, size, square_length)
        world_x, world_y = convertCoordinate(img_centerx, img_centery, size)
        return rect, box, roi, world_x, world_y

    def update_color_vote(self, observed_color):
        code = self._color_to_code(observed_color)
        if code == 0:
            return  # ignore unknown

        self.color_votes.append(code)

        if len(self.color_votes) >= self.vote_len:
            voted_code = int(round(float(np.mean(np.array(self.color_votes)))))
            self.color_votes = []
            self.confirmed_color = self._code_to_color(voted_code)
            self.draw_color = range_rgb.get(self.confirmed_color, range_rgb.get("black", (0, 0, 0)))

    def update_stability(self, rect, world_x, world_y):
        if self.last_x is None:
            self.last_x, self.last_y = world_x, world_y
            self.t_stable_start = time.time()
            self.center_list = [world_x, world_y]
            self.count = 1
            return False

        distance = math.sqrt((world_x - self.last_x) ** 2 + (world_y - self.last_y) ** 2)
        self.last_x, self.last_y = world_x, world_y

        if distance < self.stable_dist:
            if self.t_stable_start is None:
                self.t_stable_start = time.time()

            self.center_list.extend([world_x, world_y])
            self.count += 1

            if (time.time() - self.t_stable_start) >= self.stable_time:
                # stable!
                self.rotation_angle = rect[2]
                pts = np.array(self.center_list).reshape(self.count, 2)
                self.world_X_avg, self.world_Y_avg = np.mean(pts, axis=0)
                return True
            return False

        # moved too much -> reset stability window
        self.t_stable_start = time.time()
        self.center_list = [world_x, world_y]
        self.count = 1
        return False

    def process(self, img_bgr):
        annotated = img_bgr.copy()
        self.ready = False

        img_resize, img_lab = self.preprocess(img_bgr)

        best_color, best_contour, best_area = self.find_best_candidate(img_lab)
        if best_contour is None or best_area < self.min_area:
            # Optional: still show last confirmed color text
            cv2.putText(
                annotated,
                f"Color: {self.confirmed_color}",
                (10, annotated.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                self.draw_color,
                2,
            )
            return annotated, None

        rect, box, roi, world_x, world_y = self.compute_pose(best_contour)

        # Update ROI for next frame
        self.roi = roi
        self.get_roi = True

        # Annotate detected box + coords (like ColorTracking)
        cv2.drawContours(annotated, [box], -1, range_rgb.get(best_color, (255, 255, 255)), 2)
        cv2.putText(
            annotated,
            f"({world_x:.1f},{world_y:.1f})",
            (min(box[0, 0], box[2, 0]), box[2, 1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            range_rgb.get(best_color, (255, 255, 255)),
            1,
        )

        # Voting update (extra capability)
        self.update_color_vote(best_color)

        # Stability update (extra capability)
        stable_now = self.update_stability(rect, world_x, world_y)

        # Decide "ready" (state machine / gating)
        if stable_now:
            self.world_x, self.world_y = world_x, world_y
            if self.require_color_confirm:
                self.ready = (self.confirmed_color in self.target_colors)
            else:
                self.ready = True

        # Always display confirmed color text
        cv2.putText(
            annotated,
            f"Color: {self.confirmed_color}",
            (10, annotated.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            self.draw_color,
            2,
        )

        result = {
            "best_color_raw": best_color,
            "confirmed_color": self.confirmed_color,
            "area": float(best_area),
            "world_x": float(world_x),
            "world_y": float(world_y),
            "rotation_angle": float(rect[2]),
            "stable": bool(stable_now),
            "ready": bool(self.ready),
            "world_X_avg": None if self.world_X_avg is None else float(self.world_X_avg),
            "world_Y_avg": None if self.world_Y_avg is None else float(self.world_Y_avg),
        }
        return annotated, result

if __name__ == '__main__':
    init()
    start()

    perception = Perception(target_colors=('red','green','blue'))

    my_camera = Camera.Camera()
    my_camera.camera_open()

    try:
        while True:
            img = my_camera.frame
            if img is None:
                time.sleep(0.01)
                continue

            annotated, detection = perception.process(img)

            # Optional: print stable detection results
            if detection and detection["stable"]:
                print("Stable:", detection)

            cv2.imshow('Perception Demo', annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or q
                break
    finally:
        my_camera.camera_close()
        cv2.destroyAllWindows()
        stop()
        exit()


