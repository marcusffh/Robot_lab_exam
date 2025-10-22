import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import localization.particle as particle
import Utils.camera as camera
import numpy as np
import time
from timeit import default_timer as timer
from Utils.CalibratedRobot import CalibratedRobot
import math
from localization.LocalizationPathing import LocalizationPathing
import random
import cv2
from Utils.LandmarkOccupancyGrid import LandmarkOccupancyGrid
from Utils.LandmarkUtils import LandmarkUtils
from Utils.robot_model import RobotModel
from Utils.robot_RRT import robot_RRT
from localization.selflocalizeGUI import SelflocalizeGUI
from Utils.Robot import Robot

# Some color constants in BGR format
CRED = (0, 0, 255)
CGREEN = (0, 255, 0)
CBLUE = (255, 0, 0)
CYELLOW = (0, 255, 255)

# Landmarks.
landmarkIDs = [1, 2, 3, 4]
landmarks = {
    1: (0.0, 0.0),  # Coordinates for landmark 1
    2: (0.0, 200.0), # Coordinates for landmark 2
    3: (200.0, 0.0), # Coordinates for landmark 3
    4: (200.0, 200.0) # Coordinates for landmark 4
}
driving_order = [1,2,3,4,1]

landmark_radius = 20

landmark_colors = [CRED, CGREEN, CBLUE, CYELLOW]

obstacleIds_detcted = []

GUI = SelflocalizeGUI(landmarkIDs, landmark_colors, landmarks)

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

def add_obstacle_to_grid(obstacleID):
    print("addded obstacle to grid")
    x_r = est_pose.getX()
    y_r = est_pose.getY()
    theta_r = est_pose.getTheta()

    # Convert to world coordinates
    x_obj = x_r + dists[i] * np.cos(theta_r + angles[i])
    y_obj = y_r + dists[i] * np.sin(theta_r + angles[i])
    print(f"{x_obj, y_obj}")

    # Add obstacle to grid
    grid_map.add_landmark(obstacleID[i],x_obj, y_obj, landmark_radius)
    grid_map.save_map(filename=f"grid{timestep}.png")

# Main program #
try:
    # Initialize particles
    num_particles = 2000
    particles = particle.initialize_particles(num_particles)

    est_pose = particle.estimate_pose(particles) # The estimate of the robots current pose
    print(f"estimated pose: {est_pose}")

    # Driving parameters
    distance = 0.0 # distance driven at this time step
    angle = 0.0 # angle turned at this timestep

    sigma_d = 15
    sigma_theta = 0.03
    sigma_d_obs = 20
    sigma_theta_obs = 0.05

    timestep = 0

    current_goal_idx = 0
    explore_steps = 16
    pre_explore_steps = 12
    explore_counter = explore_steps
    landmarks_seen_last_timestep = []
    object_detected = False
    state = "pre_explore"

    #Initialize the robot
    arlo = CalibratedRobot()
    # Allocate space for world map
    world = np.zeros((500,500,3), dtype=np.uint8)
    # Draw map
    GUI.draw_world(est_pose, particles, world)

    #initialize helper modules    
    cam = camera.Camera(1, robottype='arlo', useCaptureThread=False)
    pathing = LocalizationPathing(arlo, landmarkIDs)
    landmark_utils = LandmarkUtils(cam, arlo)
    grid_map = LandmarkOccupancyGrid(low=(-120,-120), high=(520, 420), res=5.0)
    robot = RobotModel()

    while True:
        timestep += 1
        if current_goal_idx >= len(driving_order):
            print("All goals reached!")
            break

        #Driving logic defined by the state
        if state == "pre_explore":
            print("Pre exploring")
            if pre_explore_steps <= 11:
                distance, angle, object_detected = pathing.explore_step(False)
            pre_explore_steps -=1
            if pre_explore_steps <= 0:
                state = "navigate"

        elif state == "steer_away_from_object":
            print("steer_away_from_object")
            distance, angle = pathing.steer_away_from_object()
            object_detected = False
            explore_counter = explore_steps
            state = "explore"

        elif state == "explore":
            if explore_counter > 0:
                goal_id = driving_order[current_goal_idx]
                prev_goal_id = driving_order[current_goal_idx - 1]
                if prev_goal_id in landmarks_seen_last_timestep:
                    current_goal_idx -= 1
                    state = "navigate"
                else:
                    distance, angle, object_detected = pathing.explore_step(False)
                    explore_counter -= 1
                    print(f"Exploring after landmark, steps left: {explore_counter}")

                if object_detected:
                    state = "steer_away_from_object"
                elif explore_counter <= 0:
                    state = "navigate"

        elif state == "navigate":
            goal_id = driving_order[current_goal_idx]
            goal = landmarks[goal_id]
            print(f"Navigating to goal {goal_id}")

            # Check if direct path is clear
            if grid_map.is_path_clear([est_pose.getX(), est_pose.getY()], [goal[0], goal[1]], r_robot=20):
                distance, angle, object_detected = pathing.move_towards_goal_step(est_pose, goal)
                if object_detected:
                    state = "steer_away_from_object"
                else:
                    current_goal_idx +=1
                    explore_counter = explore_steps
                    state = "explore"
            else:
                distance, angle = 0, 0
                print("Path blocked by obstacle, using RRT")
                rrt = robot_RRT(
                    start=[est_pose.getX(), est_pose.getY()],
                    goal=[goal[0], goal[1]],
                    robot_model=robot,
                    map=grid_map,
                )
                path = rrt.planning()
                if path is not None:
                    smooth_path = rrt.smooth_path(path)
                    rrt.draw_graph(smooth_path)
                    moves, object_detected = arlo.follow_path(smooth_path)

                    for dist, ang in moves:
                        particle.sample_motion_model(particles, dist, ang, sigma_d, sigma_theta)

                    if object_detected:
                        state = "steer_away_from_object"
                    else:
                        explore_counter = explore_steps
                        state = "explore"
                else:
                    print("RRT failed to find path.")
                    state = "explore"
                
        particle.sample_motion_model(particles, distance, angle, sigma_d, sigma_theta)
        # Fetch next frame
        colour = cam.get_next_frame()
        landmarks_seen_last_timestep.clear()
        # Detect objects
        objectIDs, dists, angles = cam.detect_aruco_objects(colour)
        if not isinstance(objectIDs, type(None)):
            objectIDs, dists, angles = filter_landmarks_by_distance(objectIDs, dists, angles)
            # List detected objects
            for i in range(len(objectIDs)):
                landmarks_seen_last_timestep.append(objectIDs[i])
                print("Object ID = ", objectIDs[i], ", Distance = ", dists[i], ", angle = ", angles[i])
                if objectIDs[i] in landmarkIDs:
                    pathing.saw_landmark(objectIDs[i])
                if objectIDs[i] > 4: 
                    if objectIDs[i] in obstacleIds_detcted:
                        grid_map.remove_landmark(objectIDs[i])
                    obstacleIds_detcted.append(objectIDs[i])
                    add_obstacle_to_grid(objectIDs)
                
            # Compute particle weights
            particle.measurement_model(particles, objectIDs, landmarkIDs, landmarks, dists, angles, sigma_d_obs, sigma_theta_obs)
            # Resampling
            particles = particle.resample_particles(particles)

            # Draw detected objects
            cam.draw_aruco_objects(colour)            
        else:
            # No observation - reset weights to uniform distribution
            for p in particles:
                p.setWeight(1.0/num_particles)
    
        est_pose = particle.estimate_pose(particles) # The estimate of the robots current pose
        particles = particle.inject_random_particles(particles, ratio=0.01)

        # Draw map
        GUI.draw_world(est_pose, particles, world)
        cv2.imwrite(f"world{timestep}.png", world)
            
finally: 
    # Make sure to clean up even if an exception occurred
    
    # Close all windows
    cv2.destroyAllWindows()

    # Clean-up capture thread
    cam.terminateCaptureThread()

