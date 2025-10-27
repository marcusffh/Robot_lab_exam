import random

class Landmark:
    """Represents a fixed point in the environment. Landmarks should stay constant in position."""
    def __init__(self, landmark_id: int, x: float, y: float, radius: float = 20.0):
        self.id = landmark_id
        self.x = x
        self.y = y
        self.radius = radius
        self.visited = False

    def mark_visited(self):
        """Mark this landmark as visited."""
        self.visited = True

    def is_visited(self):
        """Check if the landmark has been visited."""
        return self.visited

class LandmarkManager:
    def __init__(self, landmarks, driving_order):
        self.landmarks = {lm.id: lm for lm in landmarks}
        self.driving_order = driving_order
        self.landmarks_seen_last_timestep = set()
    
    def get_all_ids(self):
        return list(self.landmarks.keys())

    def get_current_goal(self):
        """Return the first landmark in the driving order that hasn't been visited."""
        for lm_id in self.driving_order:
            if not self.landmarks[lm_id].is_visited():
                return self.landmarks[lm_id]
        return None
    
    def get_current_goal_position(self):
        current_goal = self.get_current_goal()
        if current_goal is not None:
            return (current_goal.x, current_goal.y)
        return None

    def mark_goal_visited(self):
        current_goal = self.get_current_goal()
        if current_goal is not None:
            current_goal.mark_visited()

    def add_landmarks_seen_last_timestep(self, landmarkIDs):
        self.landmarks_seen_last_timestep.update(landmarkIDs)

    def clear_landmarks_seen_last_timestep(self):
        self.landmarks_seen_last_timestep.clear()
    
    def current_goal_seen_last_timestep(self):
        current_goal = self.get_current_goal()
        if current_goal is None:
            return False
        return current_goal.id in self.landmarks_seen_last_timestep





