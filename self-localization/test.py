# test_rotate.py
from Utils.CalibratedRobot import CalibratedRobot
import time

# Initialize robot
robot = CalibratedRobot()

# Rotate 45 degrees
angle_deg = 90
robot.turn_angle(angle_deg)

print(f"Robot rotated {angle_deg} degrees")
time.sleep(0.5)
