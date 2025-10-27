import numpy as np
import heapq
import matplotlib.pyplot as plt
import numpy as np

class AStar:
    def __init__(self, 
                 map, 
                 r_model,
                 start,
                 goal,
                 initial_heading):

        self.map = map
        self.robot_model = r_model
        self.r_robot = r_model.robot_radius
        self.start = start
        self.goal = goal
        self.initial_heading = initial_heading

    def plan(self):
        # Convert world start/goal positions to grid indices
        start_idx, valid_start = self.map.world_to_grid(self.start)
        goal_idx, valid_goal = self.map.world_to_grid(self.goal)
        if not valid_start or not valid_goal:
            print("Start or goal is not a position on the map")
            return None

        open_list = []
        g = {tuple(start_idx): 0}
        f_start = g[tuple(start_idx)] + self.heuristic(start_idx, goal_idx)
        heapq.heappush(open_list, (f_start, tuple(start_idx)))

        came_from = {}
        visited = set()

        while open_list:
            _, s = heapq.heappop(open_list)
            visited.add(s)

            # Check goal condition
            if s == tuple(goal_idx):
                self.save_path_image()
                return self.reconstruct_path(came_from, s)

            # Get all valid neighbor nodes
            neighbors = self.get_neighbors(s, self.initial_heading if s == tuple(start_idx) else None)

            for s_prime in neighbors:
                if s_prime in visited:
                    continue

                # Movement cost from s → s'
                c = np.linalg.norm(np.array(s_prime) - np.array(s))
                g_through_s = g[s] + c

                if s_prime not in g or g_through_s < g[s_prime]:
                    g[s_prime] = g_through_s
                    f = g_through_s + self.heuristic(s_prime, goal_idx)
                    heapq.heappush(open_list, (f, s_prime))
                    came_from[s_prime] = s

        # No path found
        return None


    def get_neighbors(self, idx, initial_heading=None):
        """Return valid neighboring grid cells (collision-free)."""
        directions = [(-1,0), (1,0), (0,-1), (0,1),
                      (-1,-1), (-1,1), (1,-1), (1,1)]
        neighbors = []

        for d in directions:
            neighbor = (idx[0] + d[0], idx[1] + d[1])
            if 0 <= neighbor[0] < self.map.n_grids[0] and 0 <= neighbor[1] < self.map.n_grids[1]:
                if initial_heading is not None:
                    heading = initial_heading
                else:
                    heading = self.robot_model.compute_heading(idx, neighbor)

                if not self.map.robot_collision(neighbor, self.r_robot, heading):
                    neighbors.append(neighbor)
        return neighbors


    def heuristic(self, a, b):
        """Euclidean distance heuristic."""
        return np.linalg.norm(np.array(a) - np.array(b))


    def reconstruct_path(self, came_from, current):
        """Reconstruct path and convert to world coordinates."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()

        return [self.map.grid_to_world(p) for p in path]


    def save_path_image(self, path, filename="astar_path.png"):
        plt.figure(figsize=(6, 6))
        self.map.draw_map(robot_radius=self.r_robot)

        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1], "-b", linewidth=2, label="A* path")

        plt.axis(self.map.extent)
        plt.grid(False)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

