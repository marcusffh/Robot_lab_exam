import sys
import os
import numpy as np
import cv2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import Utils.camera as camera
from localization.selflocalizeGUI import SelflocalizeGUI # drawing the markers
from Utils.CalibratedRobot import CalibratedRobot
from Utils.LandmarkOccupancyGrid import LandmarkOccupancyGrid
from Utils.LandmarkUtils import LandmarkUtils
from Utils.robot_model import RobotModel
from Utils.robot_RRT import robot_RRT
from Utils.particle import Particle # particles and particle filter
from Utils.marker import MarkerTracker # markers and  keeping track of the

# Flags
showGUI = True  # Whether or not to open GUI windows
onRobot = True # Whether or not we are running on the Arlo robot


# Some color constants in BGR format for visualizing the map
CRED = (0, 0, 255)
CGREEN = (0, 255, 0)
CBLUE = (255, 0, 0)
CYELLOW = (0, 255, 255)

def isRunningOnArlo():
    """Return True if we are running on Arlo, otherwise False.
    You can use this flag to switch the code from running on you laptop to Arlo
    """
    return onRobot


#---------------------------------
#ARGUMENTS
num_particles = 1000

#for prediction
sigma_d_pred = 15
sigma_theta_pred = 0.03

#for Correction
sigma_d_obs = 20
sigma_theta_obs = 0.05

#rejuvination ratio
rejuvenation_ratio = 0.03

#Map bounds (for rejuvination step)
map_bounds= (640 , 540 , 0 ,0)  #  (x_max, y_max , x_min, y_min)

#---------------------------------
#INITIALIZE

# Markers and tracking
tracker = MarkerTracker()

#known makers(coordinates are in cm)
tracker.add_marker(1, 0, 0, "marker")
tracker.add_marker(2, 0, 400, "marker")
tracker.add_marker(3, 400, 0, "marker")
tracker.add_marker(4, 400, 400, "marker")

#Camera
print("Opening and initializing camera")
if isRunningOnArlo():
    cam = camera.Camera(1, robottype='arlo', useCaptureThread=False) # opens cam


# Order of visiting landmark
visiting_order = [1,2,3,4,1]


## will need------------



#------------------------



try: 
    #Initialize the robot
    if isRunningOnArlo():
        arlo = CalibratedRobot()

    particles = Particle.initialize_particles(num_particles)
    pose_estimate = Particle.estimate_pose(particles)



    while True:
        #IF All markers are NOT visited

            #look around 360
            #drive to marker in order
        
            Particle.prediction_step(particles, distance, angle_change, sigma_d = sigma_d_pred, sigma_theta=  sigma_theta_pred)

            #Get next fram

            Particle.correction_step(particles, ids, dists, angles,LANDMARKS, sigma_d = sigma_d_obs, sigma_theta = sigma_theta_obs)
            particles = Particle.resampling_step(particles)
            pose_estimate = Particle.estimate_pose(particles)
            particles = Particle.rejuvenation_step(particles, rejuvenation_ratio = 0.05, map_bounds= (520 , 420 , -120 ,- 120))
