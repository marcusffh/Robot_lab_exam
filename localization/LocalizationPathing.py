# explore_landmarks.py
from Utils.CalibratedRobot import CalibratedRobot
import time
import numpy as np

import time

class LocalizationPathing:
    def __init__(self, robot, required_landmarks, step_cm=20, rotation_deg=20, min_landmarks_to_see = 2):
        self.robot = robot
        self.required_landmarks = set(required_landmarks)
        self.step_cm = step_cm
        self.rotation_deg = rotation_deg
        self.min_landmarks_to_see = min_landmarks_to_see

        self.observed_landmarks = set()
        self.min_landmarks_met = False

    def explore_step(self, drive=False, min_dist = 400):
        dist = 0
        angle_deg = self.rotation_deg 
        angle_rad = np.radians(angle_deg)

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
        
        move_dist = distance_to_goal - 12.5
        angle_to_goal = (angle_to_goal + np.pi) % (2 * np.pi) - np.pi
        
        print(f"distance moved: {distance_to_goal}")
        print(f"angle (rad) turned: {angle_to_goal}")

        self.robot.turn_angle(np.degrees(angle_to_goal))

        distance, object_detected = self.robot.drive_distance_cm(move_dist)
        self.robot.stop()
        time.sleep(0.2)

        return distance, angle_to_goal, object_detected


def steer_away_from_object(self, turn_angle=45, stop_threshold=25):
    left, center, right = self.proximity_check()
    angle_turned = 0
    distance_moved = 0

    # If any sensor is below threshold, decide which way to turn
    if left < stop_threshold or center < stop_threshold or right < stop_threshold:
        if left > right:
            self.turn_angle(turn_angle)  # turn left
            angle_turned = np.radians(turn_angle)
        else:
            self.turn_angle(-turn_angle)  # turn right
            angle_turned = np.radians(-turn_angle)

    return distance_moved, angle_turned

