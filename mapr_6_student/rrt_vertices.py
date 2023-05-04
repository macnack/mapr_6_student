import rclpy
import time
from mapr_6_student.grid_map import GridMap
import numpy as np

np.random.seed(444)


class RRT(GridMap):
    def __init__(self):
        super(RRT, self).__init__()
        self.step = 0.05

    def check_if_valid(self, a, b):
        """
        Checks if the segment connecting a and b lies in the free space.
        :param a: point in 2D
        :param b: point in 2D
        :return: boolean
        """
        in_free_space = True
        pt_b = np.array([b[0], b[1]])
        pt_a = np.array([a[0], a[1]])
        steps = np.linspace(pt_b, pt_a, num=100)
        height, width = self.map.shape
        for step in steps:
            step *= 1.0 / self.resolution
            if step[0] < 0 or step[1] < 0 or step[1] > height or step[0] > width:
                return False
            if self.map[int(step[1]), int(step[0])] == 100:
                in_free_space = False
                break
        return in_free_space

    def random_point(self):
        """
        Draws random point in 2D
        :return: point in 2D
        """
        x = np.random.random(1) * self.width
        # x = np.around(x, decimals=1)
        y = np.random.random(1) * self.height
        # y = np.around(y, decimals=1)
        return (x[0], y[0])

    def find_closest(self, pos):
        """
        Finds the closest vertex in the graph to the pos argument

        :param pos: point id 2D
        :return: vertex from graph in 2D closest to the pos
        """
        distance = []
        childs = []
        pos = np.array([pos[0], pos[1]])
        for child in self.parent.keys():
            delta = np.array([child[0], child[1]]) - pos
            distance.append(np.linalg.norm(delta))
            childs.append(child)
        return childs[np.argmin(distance)]

    def new_pt(self, pt, closest):
        """
        Finds the point on the segment connecting closest with pt, which lies self.step from the closest (vertex in graph)

        :param pt: point in 2D
        :param closest: vertex in the tree (point in 2D)
        :return: point in 2D
        """
        closest_ = np.array([closest[0], closest[1]])
        pt_ = np.array([pt[0], pt[1]])
        delta = pt_ - closest_
        norms = np.linalg.norm(delta)
        direction = delta / norms
        new = closest_ + direction * min(self.step, norms)
        return (new[0], new[1])

    def search(self):
        """
        RRT search algorithm for start point self.start and desired state self.end.
        Saves the search tree in the self.parent dictionary, with key value pairs representing segments
        (key is the child vertex, and value is its parent vertex).
        Uses self.publish_search() and self.publish_path(path) to publish the search tree and the final path respectively.
        """
        self.parent[self.start] = None
        while True:
            x_rand = self.random_point()
            x_near = self.find_closest(x_rand)
            x_new = self.new_pt(x_rand, x_near)
            if self.check_if_valid(x_new, x_near):
                self.parent[x_new] = x_near
                self.publish_search()
                if self.check_if_valid(x_new, self.end):
                    self.parent[self.end] = x_new
                    cur_n = self.end
                    path = []
                    while cur_n is not None:
                        path.append(cur_n)
                        cur_n = self.parent[cur_n]
                    print(path)
                    self.publish_path(path)
                    break


def main(args=None):
    rclpy.init(args=args)
    rrt = RRT()
    while not rrt.data_received():
        rrt.get_logger().info("Waiting for data...")
        rclpy.spin_once(rrt)
        time.sleep(0.5)

    rrt.get_logger().info("Start graph searching!")
    time.sleep(1)
    rrt.search()


if __name__ == '__main__':
    main()
