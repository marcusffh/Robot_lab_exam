import time
from Utils.CalibratedRobot import CalibratedRobot  # adjust the import to match your file

# Initialize the robot
arlo = CalibratedRobot()

# Turn 90 degrees left
arlo.turn_angle(360)

arlo.stop()