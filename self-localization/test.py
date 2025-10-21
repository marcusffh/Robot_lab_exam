# test_move_forward.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import time

# test_rotate.py
from Utils.CalibratedRobot import CalibratedRobot
import time

# Initialize robot
robot = CalibratedRobot()

# Rotate 45 degrees
angle_deg = 360
robot.turn_angle(angle_deg)

print(f"Robot rotated {angle_deg} degrees")
time.sleep(0.5)
