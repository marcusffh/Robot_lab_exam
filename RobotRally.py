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
from Utils.AStar import AStar
from localization.selflocalizeGUI import SelflocalizeGUI
from Utils.Robot import Robot
from Utils.Landmark import Landmark, LandmarkManager

landmarks = [
    Landmark(1, 0.0, 0.0, 20.0),
    Landmark(2, 0.0, 180.0, 20.0),
    Landmark(3, 180.0, 0.0, 20.0),
    Landmark(4, 180.0, 180.0, 20.0),
]

driving_order = [1,2,3,4,1]
landmark_manager = LandmarkManager(landmarks, driving_order)
obstacleIds_detcted = []

GUI = SelflocalizeGUI(landmarks)

def filter_objects_by_distance(objectIDs, dists, angles):
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

def add_obstacle_to_grid(grid_map, obstacle_id, est_pose, dist, angle, landmark_radius, timestep):
    # Convert to world coordinates
    x_obj = est_pose.getX() + dist * np.cos(est_pose.getTheta() + angle)
    y_obj = est_pose.getY() + dist * np.sin(est_pose.getTheta() + angle)
    
    # Add obstacle to grid and save
    grid_map.add_landmark(obstacle_id, x_obj, y_obj, landmark_radius)
    grid_map.save_map(filename=f"grid{timestep}.png")

# Main program #
try:
    # Initialize particles
    num_particles = 2000
    particles = particle.initialize_particles(num_particles)

    est_pose = particle.estimate_pose(particles) # The estimate of the robots current pose
    print(f"estimated pose: {est_pose.getX(), est_pose.getY(), est_pose.getTheta()}")

    # Driving parameters
    distance = 0.0 # distance driven at this time step
    angle = 0.0 #angle turned at this timestep

    sigma_d = 10 #distance noise for motion model
    sigma_theta = 0.07 #angle noise for motion model
    sigma_d_obs = 20 #distance noise for observational model
    sigma_theta_obs = 0.1 #angle noise for observational model

    timestep = 0    

    object_detected = False # object detected with proximity sensor?
    state = "explore"

    #Initialize the robot
    arlo = CalibratedRobot()
    # Allocate space for world map
    world = np.zeros((500,500,3), dtype=np.uint8)
    # Draw map
    GUI.draw_world(est_pose, particles, world)

    #initialize helper modules    
    cam = camera.Camera(1, robottype='arlo', useCaptureThread=False)
    pathing = LocalizationPathing(arlo, cam, landmark_manager.get_all_ids())
    landmark_utils = LandmarkUtils(cam, arlo)
    grid_map = LandmarkOccupancyGrid(low=(-120,-120), high=(520, 420), res=5.0)
    robot = RobotModel()

    check_for_landmark_steps = 16
    check_for_landmark_counter = check_for_landmark_steps

    firstLandmark = True
    while True:
        timestep += 1
        if landmark_manager.get_current_goal() is None:
            print("All goals reached!")
            break

        goal_position = landmark_manager.get_current_goal_position()

        #Driving logic defined by the state
        if state == "explore":
            print("Exploring")
            if pathing.seen_enough_landmarks():
                print("seen enough landmarks")
                state = "navigate"
            else:
                distance, angle, object_detected = pathing.explore_step(False)

            if object_detected:
                state = "steer_away_from_object"

        elif state == "steer_away_from_object":
            distance, angle = pathing.steer_away_from_object()
            print(f"steering away from object, moved distance {distance}, angle {angle}")
            object_detected = False
            pathing.observed_landmarks.clear()
            pathing.min_landmarks_met = False
            state = "explore"

        elif state == "navigate":
            print(f"navigating to goal{landmark_manager.get_current_goal().id}")
            # Check if direct path is clear
            start_idx, valid = grid_map.world_to_grid([est_pose.getX(), est_pose.getY()])
            if valid:
                if grid_map.is_path_clear([est_pose.getX(), est_pose.getY()], [goal_position[0], goal_position[1]], r_robot=20):
                    print(f"path clear. est_pose: {est_pose.getX(), est_pose.getY(), est_pose.getTheta()}")
                    angle = pathing.look_towards_goal(est_pose, goal_position)
                    if pathing.sees_landmark(landmark_manager.get_current_goal().id):
                        distance, object_detected = pathing.drive_towards_goal_step(est_pose, goal_position)
                        print(f"sees goal, driving distance{distance}")
                        state = "check_if_at_landmark"
                    else:
                        particles = particle.inject_random_particles(particles, ratio=0.5)
                        print("We are lost, need to figure out where we are")
                        pathing.observed_landmarks.clear()
                        pathing.min_landmarks_met = False
                        state = "explore"

                    if object_detected:
                        state = "steer_away_from_object"
                    else:
                        state = "explore"
                else:
                    distance, angle = 0, 0
                    print("Path blocked by obstacle, using AStar")
                    a_Star = AStar(
                        map=grid_map, 
                        r_model=robot,
                        start=[est_pose.getX(), est_pose.getY()],
                        goal=[goal_position[0], goal_position[1]],
                        initial_heading=est_pose.getTheta()
                    )
                    path = a_Star.plan()
                    if path is not None:
                        moves, object_detected = arlo.follow_path(path)

                        for dist, ang in moves:
                            particle.sample_motion_model(particles, dist, ang, sigma_d, sigma_theta)

                        if object_detected:
                            state = "steer_away_from_object"
                        else:
                            state = "explore"
                    else:
                        print("Astar failed to find path.")
                        dist, angle, obstacleIds_detcted = pathing.explore_step(True)
                        state = "explore"
            else:
                particles = particle.inject_random_particles(particles, ratio=0.5)
                print("We are lost, need to figure out where we are")
                pathing.observed_landmarks.clear()
                pathing.min_landmarks_met = False
                state = "explore"

        elif state == "check_if_at_landmark":
            # Get current estimated position
            x = est_pose.getX()
            y = est_pose.getY()

            # Compute distance to goal landmark
            dist_to_landmark = np.sqrt((x - goal_position[0])**2 + (y - goal_position[1])**2)

            if dist_to_landmark <= 40:
                landmark_manager.mark_goal_visited()
                print("landmark visited")
                state = "explore"
            else:
                particles = particle.inject_random_particles(particles, ratio=0.5)
                print("We are lost, need to figure out where we are")
                pathing.observed_landmarks.clear()
                pathing.min_landmarks_met = False
                state = "explore"

                    
        particle.sample_motion_model(particles, distance, angle, sigma_d, sigma_theta)
        # Fetch next frame
        colour = cam.get_next_frame()
        landmark_manager.clear_landmarks_seen_last_timestep()
        # Detect objects
        objectIDs, dists, angles = cam.detect_aruco_objects(colour)
        if not isinstance(objectIDs, type(None)):
            objectIDs, dists, angles = filter_objects_by_distance(objectIDs, dists, angles)
            # List detected objects
            for i in range(len(objectIDs)):
                print("Object ID = ", objectIDs[i], ", Distance = ", dists[i], ", angle = ", angles[i])
                if objectIDs[i] in landmark_manager.get_all_ids():
                    landmark_manager.add_landmarks_seen_last_timestep([objectIDs[i]])
                    pathing.saw_landmark(objectIDs[i])
                if objectIDs[i] > 4: 
                    if objectIDs[i] in obstacleIds_detcted:
                        grid_map.remove_landmark(objectIDs[i])
                    obstacleIds_detcted.append(objectIDs[i])
                    add_obstacle_to_grid(grid_map, objectIDs[i],est_pose, dists[i], angles[i], 20, timestep )
                
            # Compute particle weights
            particle.measurement_model(particles, objectIDs, landmark_manager, dists, angles, sigma_d_obs, sigma_theta_obs)
            # Resampling
            particles = particle.resample_particles(particles)

            # Draw detected objects
            cam.draw_aruco_objects(colour)            
        else:
            # No observation - reset weights to uniform distribution
            for p in particles:
                p.setWeight(1.0/num_particles)
    
        est_pose = particle.estimate_pose(particles) # The estimate of the robots current pose

        # Draw map
        GUI.draw_world(est_pose, particles, world)
        cv2.imwrite(f"world{timestep}.png", world)
            
finally: 
    # Make sure to clean up even if an exception occurred
    
    # Close all windows
    cv2.destroyAllWindows()

    # Clean-up capture thread
    cam.terminateCaptureThread()

