import random

class Landmark:
    """Represents a fixed point in the environment. Landmarks should stay constant in position."""
    def __init__(self, landmark_id: int, x: float, y: float, radius: float = 20.0):
        self.id = landmark_id
        self.x = x
        self.y = y
        self.radius = radius

class LandmarkManager:
    def __init__(self, landmarks, driving_order):
        self.landmarks = {lm.id: lm for lm in landmarks}
        self.driving_order = driving_order
        self.landmarks_seen_last_timestep = set()
        self.current_index = 0
    
    def get_all_ids(self):
        return list(self.landmarks.keys())

    def get_current_goal(self):
        if self.current_index >= len(self.driving_order):
            return None
        lm_id = self.driving_order[self.current_index]
        return self.landmarks[lm_id]
    
    def get_current_goal_position(self):
        current_goal = self.get_current_goal()
        if current_goal is not None:
            return (current_goal.x, current_goal.y)
        return None

    def mark_goal_visited(self):
        if self.current_index < len(self.driving_order):
            self.current_index += 1

    def add_landmarks_seen_last_timestep(self, landmarkIDs):
        self.landmarks_seen_last_timestep.update(landmarkIDs)

    def clear_landmarks_seen_last_timestep(self):
        self.landmarks_seen_last_timestep.clear()
    
    def current_goal_seen_last_timestep(self):
        current_goal = self.get_current_goal()
        if current_goal is None:
            return False
        return current_goal.id in self.landmarks_seen_last_timestep






