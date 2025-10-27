# explore_landmarks.py
from Utils.CalibratedRobot import CalibratedRobot
import time
import numpy as np

import time

class LocalizationPathing:
    def __init__(self, robot, camera, required_landmarks, step_cm=20, rotation_deg=20, min_landmarks_to_see = 2):
        self.robot = robot
        self.required_landmarks = set(required_landmarks)
        self.step_cm = step_cm
        self.rotation_deg = rotation_deg
        self.min_landmarks_to_see = min_landmarks_to_see

        self.observed_landmarks = set()
        self.min_landmarks_met = False
        self.camera = camera

    def explore_step(self, drive=False, min_dist = 400):
        dist = 0
        angle_deg = self.rotation_deg 
        angle_rad = np.radians(angle_deg)
        object_detected = False

        if not drive:
            self.robot.turn_angle(angle_deg)
            angle_rad = np.radians(self.rotation_deg)
            time.sleep(0.2)

        if drive:
            dist = self.step_cm
            left, center, right = self.robot.proximity_check()

            if left < min_dist or center < min_dist or right < min_dist:
                self.robot.stop()
            if left > right:
                self.robot.turn_angle(45)   
                angle_rad = np.radians(45)
            else:
                self.robot.turn_angle(-45)
                angle_rad = np.radians(-45)

            dist, object_detected = self.robot.drive_distance_cm(dist)

        return dist, angle_rad, object_detected
    
    def saw_landmark(self, landmarkID):
        """
        Register that a landmark with a given ID has been seen.
        """
        if landmarkID in self.required_landmarks:
            self.observed_landmarks.add(landmarkID)

            if len(self.observed_landmarks) >= self.min_landmarks_to_see:
                self.min_landmarks_met = True

    def seen_enough_landmarks(self):
        """
        Returns True if at least `min_landmarks_seen` have been observed.
        """
        return self.min_landmarks_met

    
    def move_towards_goal_step(self, est_pose, goal):
        robot_pos = np.array([est_pose.getX(), est_pose.getY()])
        direction = goal - robot_pos
        distance_to_goal = np.linalg.norm(direction)
        angle_to_goal = np.arctan2(direction[1], direction[0]) - est_pose.getTheta()

        object_detected = False
        
        move_dist = distance_to_goal - 12.5
        angle_to_goal = (angle_to_goal + np.pi) % (2 * np.pi) - np.pi
        
        print(f"distance moved: {distance_to_goal}")
        print(f"angle (rad) turned: {angle_to_goal}")

        self.robot.turn_angle(np.degrees(angle_to_goal))

        distance, object_detected = self.robot.drive_distance_cm(move_dist)
        self.robot.stop()
        time.sleep(0.2)

        return distance, angle_to_goal, object_detected
    
    def look_towards_goal(self, est_pose, goal):
        robot_pos = np.array([est_pose.getX(), est_pose.getY()])
        direction = goal - robot_pos
        angle_to_goal = np.arctan2(direction[1], direction[0]) - est_pose.getTheta()

        angle_to_goal = (angle_to_goal + np.pi) % (2 * np.pi) - np.pi
        
        print(f"angle (rad) turned: {angle_to_goal}")

        self.robot.turn_angle(np.degrees(angle_to_goal))

        self.robot.stop()
        time.sleep(0.2)

        return angle_to_goal
    
        
    def drive_towards_goal_step(self, est_pose, goal):
        robot_pos = np.array([est_pose.getX(), est_pose.getY()])
        direction = goal - robot_pos
        distance_to_goal = np.linalg.norm(direction)

        object_detected = False
        
        move_dist = distance_to_goal - 12.5
        
        print(f"distance moved: {distance_to_goal}")

        distance, object_detected = self.robot.drive_distance_cm(move_dist)
        self.robot.stop()
        time.sleep(0.2)

        return distance, object_detected

    def steer_away_from_object(self, turn_angle=5, distance = 20, stop_threshold=210):
        left, center, right = self.robot.proximity_check()
        angle_turned = 0

        # If any sensor is below threshold, decide which way to turn
        while left < stop_threshold or center < stop_threshold or right < stop_threshold:
            if left > right:
                self.robot.turn_angle(turn_angle)  # turn left
                angle_turned += np.radians(turn_angle)
            else:
                self.robot.turn_angle(-turn_angle)  # turn right
                angle_turned -= np.radians(turn_angle)

            left, center, right = self.robot.proximity_check()

        self.robot.drive_distance_cm(distance)

        return distance, angle_turned
    
    def sees_landmark(self, landmarkId, est_pose, goal, fov=np.pi/6, distance_tolerance=50):
        """
        Check if a landmark is seen roughly in front of the robot and at roughly the correct distance.
        fov: field of view in radians (half-angle to each side)
        distance_tolerance: allowed deviation in distance (same units as robot/world coordinates)
        """
        robot_pos = np.array([est_pose.getX(), est_pose.getY()])
        goal = np.array(goal)
        distance_to_goal = np.linalg.norm(goal - robot_pos)

        colour = self.camera.get_next_frame()
        objectIDs, dists, angles = self.camera.detect_aruco_objects(colour)
        
        if objectIDs is not None and angles is not None:
            for id, d, angle in zip(objectIDs, dists, angles):
                if id == landmarkId and -fov <= angle <= fov:
                    if abs(d - distance_to_goal) <= distance_tolerance:
                        return True
        return False



