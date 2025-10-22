import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import localization.particle as particle
import Utils.camera as camera
import numpy as np
import time
from timeit import default_timer as timer
from Utils.CalibratedRobot import CalibratedRobot
from scipy.stats import norm
import math
from localization.LocalizationPathing import LocalizationPathing
import random
import cv2
from Utils.LandmarkOccupancyGrid import LandmarkOccupancyGrid
from Utils.LandmarkUtils import LandmarkUtils
from Utils.robot_model import RobotModel
from Utils.robot_RRT import robot_RRT
from localization.selflocalizeGUI import SelflocalizeGUI

# Flags
showGUI = True  # Whether or not to open GUI windows
onRobot = True # Whether or not we are running on the Arlo robot


def isRunningOnArlo():
    """Return True if we are running on Arlo, otherwise False.
    You can use this flag to switch the code from running on you laptop to Arlo - you need to do the programming here!
    """
    return onRobot

try:
    from Utils.Robot import Robot
except ImportError:
    print("selflocalize.py: robot module not present - forcing not running on Arlo!")
    onRobot = False

# Some color constants in BGR format
CRED = (0, 0, 255)
CGREEN = (0, 255, 0)
CBLUE = (255, 0, 0)
CYELLOW = (0, 255, 255)

# Landmarks.
# The robot knows the position of 2 landmarks. Their coordinates are in the unit centimeters [cm].
landmarkIDs = [1, 2, 3, 4]
landmarks = {
    1: (0.0, 0.0),  # Coordinates for landmark 1
    2: (0.0, 150.0), # Coordinates for landmark 2
    3: (200.0, 0.0), # Coordinates for landmark 3
    4: (200.0, 150.0) # Coordinates for landmark 4
}

offset = 10.0
goals = {
    1: (0.0 + offset, 0.0 + offset),
    2: (0.0 + offset, 150.0 - offset),
    3: (200.0 - offset, 0.0 + offset),
    4: (200.0 - offset, 150.0 - offset)
}

landmark_order = [1,2,3,4,1]

landmark_radius = 20

landmark_colors = [CRED, CGREEN, CBLUE, CYELLOW]

obstacleIds_detcted = []

GUI = SelflocalizeGUI(landmarkIDs, landmark_colors, landmarks)

def initialize_particles(num_particles):
    particles = []
    for i in range(num_particles):
        # Random starting points. 
        p = particle.Particle(520.0*np.random.ranf() - 120.0, 420.0*np.random.ranf() - 120.0, np.mod(2.0*np.pi*np.random.ranf(), 2.0*np.pi), 1.0/num_particles)
        particles.append(p)

    return particles

def sample_motion_model(particles_list, distance, angle, sigma_d, sigma_theta):
    for p in particles_list:
        delta_x = distance * np.cos(p.getTheta() + angle)
        delta_y = distance * np.sin(p.getTheta() + angle)
    
        particle.move_particle(p, delta_x, delta_y, angle)
    if not(distance == 0 and angle == 0):
        particle.add_uncertainty(particles_list, sigma_d, sigma_theta)


def measurement_model(particle_list, ObjectIDs, dists, angles, sigma_d, sigma_theta):
    for particle in particle_list:
        x_i = particle.getX()
        y_i = particle.getY()
        theta_i = particle.getTheta()

        p_observation_given_x = 1.0

        #p(z|x) = sum over the probability for all landmarks
        for landmarkID, dist, angle in zip(ObjectIDs, dists, angles):
            if landmarkID in landmarkIDs:
                l_x, l_y = landmarks[landmarkID]
                d_i = np.sqrt((l_x - x_i)**2 + (l_y - y_i)**2)

                p_d_m = norm.pdf(dist, loc=d_i, scale=sigma_d)

                e_theta = np.array([np.cos(theta_i), np.sin(theta_i)])
                e_theta_hat = np.array([-np.sin(theta_i), np.cos(theta_i)])

                e_l = np.array([l_x - x_i, l_y - y_i]) / d_i

                dot = np.clip(np.dot(e_l, e_theta), -1.0, 1.0)
                phi_i = np.sign(np.dot(e_l, e_theta_hat)) * np.arccos(dot)
                
                p_phi_m = norm.pdf(angle,loc=phi_i, scale=sigma_theta)


                p_observation_given_x *= p_d_m* p_phi_m

        particle.setWeight(p_observation_given_x)


def resample_particles(particle_list):
    weights = np.array([p.getWeight() for p in particle_list])
    total_weight = np.sum(weights)

    # Avoid divide-by-zero
    if total_weight == 0 or np.isnan(total_weight):
        weights = np.ones(len(particle_list)) / len(particle_list)
    else:
        weights /= total_weight

    cdf = np.cumsum(weights)
    resampled = []

    for _ in range(len(particle_list)):
        z = np.random.rand()
        idx = np.searchsorted(cdf, z)
        p_resampled = particle.Particle(
            particle_list[idx].getX(),
            particle_list[idx].getY(),
            particle_list[idx].getTheta(),
            1.0 / len(particle_list)
        )
        resampled.append(p_resampled)

    rejuvenation_ratio = 0.05  # 5% random new particles
    n_random = int(len(particle_list) * rejuvenation_ratio)

    for i in range(n_random):
        resampled[i] = particle.Particle(
            520.0 * np.random.ranf() - 120.0,
            420.0 * np.random.ranf() - 120.0,
            np.mod(2.0 * np.pi * np.random.ranf(), 2.0 * np.pi),
            1.0 / len(particle_list)
        )

    return resampled


def filter_landmarks_by_distance(objectIDs, dists, angles):
    """
    Keep only the measurement at the smallest distance for each landmark ID.
    """
    min_dist_dict = {}  # dict: landmarkID -> (dist, angle)

    for id, d, a in zip(objectIDs, dists, angles):
        if id not in min_dist_dict or d < min_dist_dict[id][0]:
            min_dist_dict[id] = (d, a)

    filtered_ids = list(min_dist_dict.keys())
    filtered_dists = [min_dist_dict[ID][0] for ID in filtered_ids]
    filtered_angles = [min_dist_dict[ID][1] for ID in filtered_ids]

    return filtered_ids, filtered_dists, filtered_angles

# Main program #
try:
    # Initialize particles
    num_particles = 1000
    particles = initialize_particles(num_particles)

    est_pose = particle.estimate_pose(particles) # The estimate of the robots current pose
    print(f"estimated pose: {est_pose}")

    # Driving parameters
    distance = 0.0 # distance driven at this time step
    angle = 0.0 # angle turned at this timestep

    sigma_d = 15
    sigma_theta = 0.03
    sigma_d_obs = 20
    sigma_theta_obs = 0.05

    counter = 0

    current_goal_idx = 0

    just_moved_to_landmark = False
    explore_steps_after_landmark = 12
    explore_counter = 0

    #Initialize the robot
    if isRunningOnArlo():
        arlo = CalibratedRobot()
    # Allocate space for world map
    world = np.zeros((500,500,3), dtype=np.uint8)

    # Draw map
    GUI.draw_world(est_pose, particles, world)

    print("Opening and initializing camera")
    if isRunningOnArlo():
        #cam = camera.Camera(0, robottype='arlo', useCaptureThread=True)
        cam = camera.Camera(1, robottype='arlo', useCaptureThread=False)
        pathing = LocalizationPathing(arlo, landmarkIDs)
        landmark_utils = LandmarkUtils(cam, arlo)
        grid_map = LandmarkOccupancyGrid(low=(-120,-120), high=(520, 420), res=5.0)
        robot = RobotModel()
    else:
        cam = camera.Camera(0, robottype='macbookpro', useCaptureThread=False)

    while True:
        if current_goal_idx > len(landmark_order):
            print("All goals reached!")
            break
        # Use motor controls to update particles
        if isRunningOnArlo():
            counter +=1
            if counter > 1:
                if not (pathing.seen_enough_landmarks()):
                    distance, angle = pathing.explore_step(False)
                    print("exploring")
                elif pathing.seen_enough_landmarks() and just_moved_to_landmark:
                    if explore_counter > 0:
                        distance, angle = pathing.explore_step(False)
                        explore_counter -= 1
                        print(f"Exploring after reached landmark, steps left: {explore_counter}")
                    else:
                        just_moved_to_landmark = False
                else:
                    goal_id = landmark_order[current_goal_idx]  
                    goal = goals[goal_id]
                    
                    #print(f"{[est_pose.getX(), est_pose.getY()], [goal[0], goal[1]], grid_map.is_path_clear([est_pose.getX(), est_pose.getY()], [goal[0], goal[1]], r_robot=20)}")
                    #if grid_map.is_path_clear([est_pose.getX(), est_pose.getY()], [goal[0], goal[1]], r_robot=20):
                    print(f"driving to_landmark {goal_id}")
                    distance, angle = pathing.move_towards_goal_step(est_pose, goal)
                    current_goal_idx +=1
                    just_moved_to_landmark = True
                    explore_counter = explore_steps_after_landmark
                    #else:
                    #    rrt = robot_RRT(
                    #        start=[est_pose.getX(), est_pose.getY()],
                    #        goal=[goal[0], goal[1]],
                    #        robot_model=robot,
                    #        map=grid_map,   
                    #        )
                    #    current_goal_idx += 1
                    #    path =rrt.planning()
                    #    smooth_path = rrt.smooth_path(path)
                    #    moves = arlo.follow_path(smooth_path)
                    #    for dist, ang in moves:
                    #        sample_motion_model(particles, dist, ang, sigma_d, sigma_theta)
                
        sample_motion_model(particles, distance, angle, sigma_d, sigma_theta)
        # Fetch next frame
        colour = cam.get_next_frame()
        
        # Detect objects
        objectIDs, dists, angles = cam.detect_aruco_objects(colour)
        if not isinstance(objectIDs, type(None)):
            objectIDs, dists, angles = filter_landmarks_by_distance(objectIDs, dists, angles)
            # List detected objects
            for i in range(len(objectIDs)):
                print("Object ID = ", objectIDs[i], ", Distance = ", dists[i], ", angle = ", angles[i])
                if objectIDs[i] in landmarkIDs:
                    pathing.saw_landlanmark(objectIDs[i])
                if objectIDs[i] > 4: 
                    if objectIDs[i] in obstacleIds_detcted:
                        grid_map.remove_landmark(objectIDs[i])
                    else: 
                        obstacleIds_detcted.append(objectIDs[i])
                        print("addded obstacle to grid")
                        x_r = est_pose.getX()
                        y_r = est_pose.getY()
                        theta_r = est_pose.getTheta()

                        # Convert to world coordinates
                        x_obj = x_r + dists[i] * np.cos(theta_r + angles[i])
                        y_obj = y_r + dists[i] * np.sin(theta_r + angles[i])

                        # Add obstacle to grid
                        grid_map.add_landmark(objectIDs[i],x_obj, y_obj, landmark_radius)
                    
            # Compute particle weights
            measurement_model(particles, objectIDs, dists, angles, sigma_d_obs, sigma_theta_obs)
            # Resampling
            particles = resample_particles(particles)

            # Draw detected objects
            cam.draw_aruco_objects(colour)            
        else:
            # No observation - reset weights to uniform distribution
            for p in particles:
                p.setWeight(1.0/num_particles)
    
        est_pose = particle.estimate_pose(particles) # The estimate of the robots current pose

        if showGUI:
            # Draw map
            GUI.draw_world(est_pose, particles, world)
            cv2.imwrite(f"world{counter}.png", world)
            #grid_map.save_map(filename=f"grid{counter}.png")

finally: 
    # Make sure to clean up even if an exception occurred
    
    # Close all windows
    cv2.destroyAllWindows()

    # Clean-up capture thread
    cam.terminateCaptureThread()

