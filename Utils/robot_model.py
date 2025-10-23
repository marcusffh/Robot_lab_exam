import numpy as np

class RobotModel:
    def __init__(self, robot_radius=0.25):
        self.robot_radius = robot_radius

    def compute_heading(self, from_idx, to_idx):
        vec = np.array(to_idx) - np.array(from_idx)
        return np.arctan2(vec[1], vec[0])
