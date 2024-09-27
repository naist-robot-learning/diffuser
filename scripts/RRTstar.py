import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from matplotlib.animation import FuncAnimation


class Node:
    def __init__(self, config, position, cost, parent=None):
        self.config = config
        self.position = position
        self.cost = cost
        self.parent = parent


class RRTStar:
    def __init__(self, start, goal, bounds, obstacles, max_iter=1000, step_size=1.0, search_radius=1.0):
        self.start = Node(config=start[0], position=start[1], cost=start[2])
        self.goal = Node(config=goal[0], position=goal[1], cost=goal[2])
        self.bounds = bounds
        self.obstacles = obstacles
        self.max_iter = max_iter
        self.step_size = step_size
        self.search_radius = search_radius
        self.tree = [self.start]
        self.kd_tree = cKDTree([self.start.position])  # K-Dimensional Tree to search for nearest neighboors
        self.fig, self.ax = plt.subplots()
        self.lines = []

    def distance(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def sample_free(self):
        while True:
            sample = (
                np.random.uniform(self.bounds[0][0], self.bounds[0][1]),
                np.random.uniform(self.bounds[1][0], self.bounds[1][1]),
            )
            if not self.in_obstacle(sample):
                return sample

    def in_obstacle(self, point):
        for ox, oy, radius in self.obstacles:
            if self.distance(point, (ox, oy)) <= radius:
                return True
        return False

    def steer(self, from_node, to_position):
        direction = np.array(to_position) - np.array(from_node.position)
        length = np.linalg.norm(direction)
        direction = direction / length
        new_position = np.array(from_node.position) + direction * min(self.step_size, length)
        new_position = tuple(new_position)

        if not self.in_obstacle(new_position):
            new_config = from_node.config  # Simplified; should include actual robot kinematics
            new_cost = from_node.cost + self.distance(from_node.position, new_position)
            return Node(config=new_config, position=new_position, cost=new_cost, parent=from_node)
        return None

    def get_nearest(self, position):
        distances, indices = self.kd_tree.query(position)
        return self.tree[indices]

    def get_near(self, position):
        indices = self.kd_tree.query_ball_point(position, self.search_radius)
        return [self.tree[i] for i in indices]

    def rewire(self, new_node, near_nodes):
        for near_node in near_nodes:
            new_cost = new_node.cost + self.distance(new_node.position, near_node.position)
            if new_cost < near_node.cost:
                near_node.parent = new_node
                near_node.cost = new_cost

    def plan(self):
        for _ in range(self.max_iter):
            random_position = self.sample_free()
            nearest_node = self.get_nearest(random_position)
            new_node = self.steer(nearest_node, random_position)

            if new_node:
                near_nodes = self.get_near(new_node.position)
                self.rewire(new_node, near_nodes)
                self.tree.append(new_node)
                self.kd_tree = cKDTree([node.position for node in self.tree])

                # Update the plot
                self.update_plot(new_node)

                if self.distance(new_node.position, self.goal.position) < self.step_size:
                    print("Goal reached!")
                    self.plot_path(new_node)
                    return self.get_path(new_node)

        print("Goal not reached within max iterations.")
        return None

    def update_plot(self, new_node):
        if new_node.parent:
            (line,) = self.ax.plot(
                [new_node.position[0], new_node.parent.position[0]],
                [new_node.position[1], new_node.parent.position[1]],
                "b-",
            )
            self.lines.append(line)
            plt.pause(0.01)

    def get_path(self, node):
        path = []
        while node:
            path.append(node.position)
            node = node.parent
        return path[::-1]

    def plot_path(self, node):
        path = self.get_path(node)
        self.ax.plot([pos[0] for pos in path], [pos[1] for pos in path], "g-", linewidth=2)

    def setup_plot(self):
        self.ax.set_xlim(self.bounds[0])
        self.ax.set_ylim(self.bounds[1])
        for ox, oy, radius in self.obstacles:
            circle = plt.Circle((ox, oy), radius, color="r")
            self.ax.add_artist(circle)
        self.ax.plot(self.start.position[0], self.start.position[1], "ro")
        self.ax.plot(self.goal.position[0], self.goal.position[1], "go")


# Example usage
start = (0, (0, 0), 0)
goal = (0, (10, 10), 0)
bounds = ((-5, 15), (-5, 15))
obstacles = [(5, 5, 1), (7, 8, 1.5)]

rrt_star = RRTStar(start, goal, bounds, obstacles)
rrt_star.setup_plot()
path = rrt_star.plan()

if path:
    plt.show()
