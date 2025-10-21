# test_move_forward.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from LocalizationPathing import LocalizationPathing
from Utils.CalibratedRobot import CalibratedRobot
import camera
import numpy as np
import time

# Initialize robot and camera
robot = CalibratedRobot()

# Dummy landmarks list (won't be used in this test)
required_landmarks = []

# Initialize LocalizationPathing
localization = LocalizationPathing(robot)

# Create a dummy estimated pose object
class DummyPose:
    def __init__(self, x=0, y=0, theta=0):
        self._x = x
        self._y = y
        self._theta = theta
    def getX(self):
        return self._x
    def getY(self):
        return self._y
    def getTheta(self):
        return self._theta

# Current pose at origin facing along x-axis
est_pose = DummyPose(0, 0, 0)

# Goal is 100 cm ahead along x-axis
goal = np.array([400, 0])

# Move towards the goal
distance, angle = localization.move_towards_goal_step(est_pose, goal)

print(f"Moved {distance} cm and turned {np.degrees(angle)} degrees")
