
class Marker:
    """Class to represent and track an ArUco marker"""
    
    def __init__(self, id, x, y, marker_type="marker", visited=False):
        """
        Initialize a marker
        
        Args:
            id (int): Unique marker ID
            x (float): X coordinate
            y (float): Y coordinate
            marker_type (str): Type of marker - "marker" or "obstacle"
            visited (bool): Whether the marker has been visited
        """
        self.id = id
        self.x = x
        self.y = y
        self.type = marker_type
        self.visited = visited
    
    def update_coordinates(self, x, y):
        """Update marker coordinates"""
        self.x = x
        self.y = y
    
    def update_type(self, marker_type):
        """Update marker type"""
        if marker_type not in ["marker", "obstacle"]:
            raise ValueError("Type must be 'marker' or 'obstacle'")
        self.type = marker_type
    
    def mark_visited(self):
        """Mark the marker as visited"""
        self.visited = True
    
    def mark_unvisited(self):
        """Mark the marker as unvisited"""
        self.visited = False
    
    def __str__(self):
        """String representation of the marker"""
        return f"Marker(ID={self.id}, pos=({self.x}, {self.y}), type={self.type}, visited={self.visited})"
    
    def __repr__(self):
        return self.__str__()



class MarkerTracker:
    """Class to manage multiple markers"""
    
    def __init__(self):
        """Initialize the marker tracker"""
        self.markers = {}
    
    def add_marker(self, id, x, y, marker_type="marker", visited=False):
        """
        Add a new marker to the tracker
        
        Args:
            id (int): Unique marker ID
            x (float): X coordinate
            y (float): Y coordinate
            marker_type (str): Type of marker - "marker" or "obstacle"
            visited (bool): Whether the marker has been visited
        
        Returns:
            Marker: The created marker object
        """
        if id in self.markers:
            print(f"Warning: Marker with ID {id} already exists. Updating instead.")
        
        marker = Marker(id, x, y, marker_type, visited)
        self.markers[id] = marker
        return marker
    
    def remove_marker(self, id):
        """
        Remove a marker by ID
        
        Args:
            id (int): Marker ID to remove
        
        Returns:
            bool: True if removed, False if not found
        """
        if id in self.markers:
            del self.markers[id]
            return True
        return False
    
    def get_marker(self, id):
        """
        Get a marker by ID
        
        Args:
            id (int): Marker ID
        
        Returns:
            Marker: The marker object, or None if not found
        """
        return self.markers.get(id)
    
    def update_marker_coordinates(self, id, x, y):
        """Update coordinates of a specific marker"""
        marker = self.get_marker(id)
        if marker:
            marker.update_coordinates(x, y)
            return True
        return False
    
    def update_marker_type(self, id, marker_type):
        """Update type of a specific marker"""
        marker = self.get_marker(id)
        if marker:
            marker.update_type(marker_type)
            return True
        return False
    
    def mark_visited(self, id):
        """Mark a marker as visited"""
        marker = self.get_marker(id)
        if marker:
            marker.mark_visited()
            return True
        return False
    
    def get_all_markers(self):
        """Get all markers as a list"""
        return list(self.markers.values())
    
    def get_visited_markers(self):
        """Get all visited markers"""
        return [m for m in self.markers.values() if m.visited]
    
    def get_unvisited_markers(self):
        """Get all unvisited markers"""
        return [m for m in self.markers.values() if not m.visited]
    
    def get_markers_by_type(self, marker_type):
        """Get all markers of a specific type"""
        return [m for m in self.markers.values() if m.type == marker_type]
    
    def __str__(self):
        return f"MarkerTracker with {len(self.markers)} markers"
