# Created by arthavnuc

import numpy as np
import heapq
import time
import cv2
from distlib.compat import raw_input
from collections import deque

# Globally defined multiplier
MULTIPLIER = 10

# A container to hold multiple time values in key: value pair to be used for analysis at the termination of program
time_dict = {}

# Defining some colors for visualization in BGR
#  obstacle space
black = (0, 0, 0)
# white denotes free space
white = (255, 255, 255)
# final path tracing
red = (0, 0, 255)
# explored nodes
grey = (128, 128, 128)
green = (0, 255, 0)


# Point container class
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def move(self, dx, dy):
        self.x += dx
        self.y += dy

    def __str__(self):
        return f"Point({self.x}, {self.y})"

# Node container class
class Node:
    def __init__(self, x, y, c2c):
        self.x = x
        self.y = y
        self.c2c = c2c
        self.parent = None
        self.visited = False
    # used for duplicate key finding in dict
    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.x == other.x and self.y == other.y
        return False

    def __lt__(self, other):
        return self.c2c < other.c2c

    def __str__(self):
        return f"Node({self.x}, {self.y}, {self.c2c})"

# class to create canvas having obstacle and boundaries
class Canvas:
    def __init__(self, width, height, clearance=2, multiplier=1):
        start_time = time.perf_counter()
        self.multiplier = multiplier
        self.width = width * self.multiplier
        self.height = height * self.multiplier
        self.clearance = clearance * self.multiplier
        # using 3D array for color visualization in opencv mat
        #  white canvas
        self.canvas = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255

        print("Preparing Canvas")
        self._draw_borders()
        self._draw_obstacles()
        end_time = time.perf_counter()
        time_dict["map creation"] = end_time - start_time

    # Function to draw borders on canvas
    def _draw_borders(self):
        cv2.rectangle(img=self.canvas, pt1=(0, 0), pt2=(self.width, self.height), color=black,
                      thickness=self.clearance)
    # Function to visualise the canvas for debugging
    def _visualize_canvas(self):
        cv2.imshow("img", self.canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Function calling each type of obstacle to be drawn
    def _draw_obstacles(self):
        self._draw_E(Point(15, 12.5), 13, 5)
        self._draw_N(Point(32, 12.5), 25, 5)
        self._draw_P(Point(53, 12.5), 6, 5)
        self._draw_M(Point(68, 12.5), 25, 5)
        self._draw_6(Point(101, 12.5), 9, 5)
        self._draw_6(Point(121, 12.5), 9, 5)
        self._draw_1(Point(145, 12.5), 5, 25)

    def _draw_1(self, topcorner, stem, height):
        stem = stem * self.multiplier
        height = height * self.multiplier
        topcorner.x = int(topcorner.x * self.multiplier)
        topcorner.y = int(topcorner.y * self.multiplier)
        pts = (Point(topcorner.x, topcorner.y),
               Point(topcorner.x + stem, topcorner.y),
               Point(topcorner.x + stem, topcorner.y + height),
               Point(topcorner.x, topcorner.y + height),)
        # padding
        for p in pts:
            p.x = p.x + self.clearance
            p.y = p.y + self.clearance
        arr = []
        list(map(lambda p: arr.append([p.x, p.y]), pts))

        cv2.fillPoly(img=self.canvas, pts=[np.array(arr).reshape(-1, 1, 2)], color=black)

    def _draw_E(self, topcorner: Point, width, stemwidth):
        width = width * self.multiplier
        stemwidth = stemwidth * self.multiplier
        topcorner.x = int(topcorner.x * self.multiplier)
        topcorner.y = int(topcorner.y * self.multiplier)

        pts = (Point(topcorner.x, topcorner.y),
               Point(topcorner.x + width, topcorner.y),
               Point(topcorner.x + width, topcorner.y + stemwidth),
               Point(topcorner.x + stemwidth, topcorner.y + stemwidth),
               Point(topcorner.x + stemwidth, topcorner.y + stemwidth + stemwidth),
               Point(topcorner.x + width, topcorner.y + stemwidth + stemwidth),
               Point(topcorner.x + width, topcorner.y + 3 * stemwidth),
               Point(topcorner.x + stemwidth, topcorner.y + 3 * stemwidth),
               Point(topcorner.x + stemwidth, topcorner.y + 4 * stemwidth),
               Point(topcorner.x + width, topcorner.y + 4 * stemwidth),
               Point(topcorner.x + width, topcorner.y + 5 * stemwidth),
               Point(topcorner.x, topcorner.y + 5 * stemwidth),)

        # padding
        for p in pts:
            p.x = p.x + self.clearance
            p.y = p.y + self.clearance
        arr = []
        list(map(lambda p: arr.append([p.x, p.y]), pts))

        cv2.fillPoly(img=self.canvas, pts=[np.array(arr).reshape(-1, 1, 2)], color=black)

    def _draw_N(self, topcorner: Point, height, stemwidth):
        height = height * self.multiplier
        stemwidth = stemwidth * self.multiplier
        topcorner.x = int(topcorner.x * self.multiplier)
        topcorner.y = int(topcorner.y * self.multiplier)

        pts = (Point(topcorner.x, topcorner.y),
               Point(topcorner.x + stemwidth, topcorner.y),
               Point(topcorner.x + stemwidth * 2, topcorner.y + 3 * stemwidth),
               Point(topcorner.x + stemwidth * 2, topcorner.y),
               Point(topcorner.x + stemwidth * 3, topcorner.y),
               Point(topcorner.x + 3 * stemwidth, topcorner.y + height),
               Point(topcorner.x + 2 * stemwidth, topcorner.y + height),
               Point(topcorner.x + stemwidth, topcorner.y + stemwidth),
               Point(topcorner.x + stemwidth, topcorner.y + height),
               Point(topcorner.x, topcorner.y + height),
               )
        # padding
        for p in pts:
            p.x = p.x + self.clearance
            p.y = p.y + self.clearance
        arr = []
        list(map(lambda p: arr.append([p.x, p.y]), pts))
        cv2.fillPoly(img=self.canvas, pts=[np.array(arr).reshape(-1, 1, 2)], color=black)

    def _draw_P(self, topcorner: Point, radius, stemwidth):
        radius = radius * self.multiplier
        stemwidth = stemwidth * self.multiplier
        topcorner.x = int(topcorner.x * self.multiplier)
        topcorner.y = int(topcorner.y * self.multiplier)
        center = Point(topcorner.x + radius + self.clearance, topcorner.y + radius + self.clearance)

        pts = (Point(topcorner.x, topcorner.y),
               Point(topcorner.x + stemwidth, topcorner.y),
               Point(topcorner.x + stemwidth, topcorner.y + 5 * stemwidth),
               Point(topcorner.x, topcorner.y + 5 * stemwidth),)

        angles = np.arange(0, 2 * np.pi, np.pi / 10)
        pts_list = []
        for idx, angle in enumerate(angles):
            pts_list.append([int(radius * np.cos(angle) + center.x), int(radius * np.sin(angle) + center.y)])

        # padding
        for p in pts:
            p.x = p.x + self.clearance
            p.y = p.y + self.clearance
        arr = []
        list(map(lambda p: arr.append([p.x, p.y]), pts))
        cv2.fillPoly(img=self.canvas, pts=[np.array(pts_list).reshape(-1, 1, 2)], color=black)
        cv2.fillPoly(img=self.canvas, pts=[np.array(arr).reshape(-1, 1, 2)], color=black)

    def _draw_M(self, topcorner: Point, height, stemwidth):
        height = height * self.multiplier
        stemwidth = stemwidth * self.multiplier
        topcorner.x = int(topcorner.x * self.multiplier)
        topcorner.y = int(topcorner.y * self.multiplier)

        pts = (Point(topcorner.x, topcorner.y),
               Point(topcorner.x + stemwidth, topcorner.y),

               Point(topcorner.x + stemwidth * 2, topcorner.y + 4 * stemwidth),

               Point(topcorner.x + stemwidth * 2 + 70, topcorner.y + 4 * stemwidth),

               Point(topcorner.x + stemwidth * 4 + 20, topcorner.y),
               Point(topcorner.x + stemwidth * 4 + 70, topcorner.y),
               Point(topcorner.x + stemwidth * 4 + 70, topcorner.y + height),
               Point(topcorner.x + stemwidth * 4 + 20, topcorner.y + height),
               Point(topcorner.x + stemwidth * 4 + 20, topcorner.y + int(height / 2)),
               Point(topcorner.x + stemwidth * 2 + 70, topcorner.y + height),
               Point(topcorner.x + stemwidth * 2, topcorner.y + height),
               Point(topcorner.x + stemwidth, topcorner.y + int(height / 2)),
               Point(topcorner.x + stemwidth, topcorner.y + height),
               Point(topcorner.x, topcorner.y + height),

               )
        # padding
        for p in pts:
            p.x = p.x + self.clearance
            p.y = p.y + self.clearance
        arr = []
        list(map(lambda p: arr.append([p.x, p.y]), pts))
        cv2.fillPoly(img=self.canvas, pts=[np.array(arr).reshape(-1, 1, 2)], color=black)

    def _draw_6(self, topcorner: Point, radius, stemwidth):
        radius = radius * self.multiplier
        stemwidth = stemwidth * self.multiplier
        topcorner.x = int(topcorner.x * self.multiplier)
        topcorner.y = int(topcorner.y * self.multiplier)
        center = Point(topcorner.x + radius + self.clearance, topcorner.y + 160 + self.clearance)
        pts = (Point(topcorner.x, topcorner.y),
               Point(topcorner.x + stemwidth, topcorner.y),
               Point(topcorner.x + stemwidth, topcorner.y + 5 * stemwidth - radius),
               Point(topcorner.x, topcorner.y + 5 * stemwidth - radius),)

        angles = np.arange(0, 2 * np.pi, np.pi / 10)
        pts_list = []
        for idx, angle in enumerate(angles):
            pts_list.append([int(radius * np.cos(angle) + center.x), int(radius * np.sin(angle) + center.y)])

        # padding
        for p in pts:
            p.x = p.x + self.clearance
            p.y = p.y + self.clearance
        arr = []
        list(map(lambda p: arr.append([p.x, p.y]), pts))
        cv2.fillPoly(img=self.canvas, pts=[np.array(pts_list).reshape(-1, 1, 2)], color=black)
        cv2.fillPoly(img=self.canvas, pts=[np.array(arr).reshape(-1, 1, 2)], color=black)


# Combined class for BFS and Dijkstra
class GraphSearch:
    def __init__(self, canvas: np.ndarray, algo: str):
        try:
            self.canvas = canvas
            self.start_pt = None
            self.end_pt = None
            self.queue = []
            self._input_start_pt()
            self._goal_start_pt()
            self.nodes_dict = {}
            # False for BFS True for Dijkstra
            if algo == "BFS":
                self.algo = False
            elif algo == "DIJKSTRA":
                self.algo = True
            else:
                raise NotImplementedError
        except NotImplementedError:
            print("Invalid Algorithm!! Please choose from BFS, DIJKSTRA")
            return

        if self.algo:
            self._dijkstra()
        else:
            self._bfs_search()

    # Loop till user inputs valid pt i.e. pt in free space in canvas (checking color to be white)
    def _input_start_pt(self):
        print("Enter the start point of the path - X{}, Y{} ".format((0,self.canvas.shape[1]),(0,self.canvas.shape[0])))
        while True:
            x, y = raw_input().split(",")
            # convert point in frame having oring on top left
            y = -int(y) + int(self.canvas.shape[0])
            p = Point(int(x), y)
            if self._pt_valid(p):
                self.start_pt = p
                break
            else:
                print("Invalid Input!! Please enter X and Y")

    # Function to input and validate the goal pt
    def _goal_start_pt(self):
        print("Enter the End point of the path - X{}, Y{} ".format((0,self.canvas.shape[1]),(0,self.canvas.shape[0])))
        while True:
            x, y = raw_input().split(",")
            # convert point in frame having oring on top left
            y = -int(y) + int(self.canvas.shape[0])
            p = Point(int(x), y)
            if self._pt_valid(p):
                self.end_pt = p
                break
            else:
                print("Invalid Input!! Please enter X and Y")

    # This function checks for validity of pt i.e. pt has to be in free space and in bounds for canvas
    def _pt_valid(self, pt: Point):
        if not (0 < pt.x < self.canvas.shape[1] and 0 < pt.y < self.canvas.shape[0]):
            print("Point out of canvas")
            return False
        elif self.canvas[pt.y, pt.x, 0] != 255:
            print("Point inside obstacle")
            return False
        else:
            return True

    # Function implementing BFS
    def _bfs_search(self):
        # action and cost in single container
        start_time = time.perf_counter()
        self.nodes_dict.clear()
        # Action list for 8 grid
        valid_actions = {(1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1), (1, 1, 1.4), (-1, 1, 1), (1, -1, 1),
                         (-1, -1, 1)}

        u = Node(self.start_pt.x, self.start_pt.y, 0)
        self.queue = deque()
        self.queue.append(u)
        goal_reached = False
        self.nodes_dict = {(u.x, u.y): u}

        while self.queue:
            node = self.queue.popleft()

            if node.x == self.end_pt.x and node.y == self.end_pt.y:
                print("Path found!")
                goal_reached = True
                break

            for dx, dy, cost in valid_actions:
                #  check in dict if exist return that else new
                next_node = self.nodes_dict.get((node.x + dx, node.y + dy), Node(node.x + dx, node.y + dy, 0))

                # ensure move is in bounds and in free space
                if 0 <= next_node.x < self.canvas.shape[1] and 0 <= next_node.y < self.canvas.shape[0] and self.canvas[
                    next_node.y, next_node.x, 0] == 255:
                    if next_node.visited == False or next_node.c2c > node.c2c + cost:
                        next_node.c2c = node.c2c + cost
                        next_node.visited = True
                        next_node.parent = node
                        self.queue.append(next_node)
                        self.nodes_dict[(next_node.x, next_node.y)] = next_node

        if not goal_reached:
            print("No path found!")
        end_time = time.perf_counter()
        time_dict["BFS"] = end_time - start_time

    # Function implementing Dijkstra
    def _dijkstra(self):
        start_time = time.perf_counter()
        self.nodes_dict.clear()
        # Action list for 8 grid
        valid_actions = {(1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1), (1, 1, 1.4), (-1, 1, 1.4), (1, -1, 1.4),
                         (-1, -1, 1.4)}

        u = Node(self.start_pt.x, self.start_pt.y, 0)
        heapq.heappush(self.queue, u)
        goal_reached = False
        self.nodes_dict = {(u.x, u.y): u}

        while self.queue:
            node = heapq.heappop(self.queue)

            if node.x == self.end_pt.x and node.y == self.end_pt.y:
                print("Path found!")
                goal_reached = True
                break

            for dx, dy, cost in valid_actions:
                #  check in dict if exist return that else new
                next_node = self.nodes_dict.get((node.x + dx, node.y + dy), Node(node.x + dx, node.y + dy, 0))

                # ensure move is in bounds and in free space
                if 0 <= next_node.x < self.canvas.shape[1] and 0 <= next_node.y < self.canvas.shape[0] and self.canvas[
                    next_node.y, next_node.x, 0] == 255:
                    if next_node.visited == False or next_node.c2c > node.c2c + cost:
                        next_node.c2c = node.c2c + cost
                        next_node.visited = True
                        next_node.parent = node
                        heapq.heappush(self.queue, next_node)
                        self.nodes_dict[(next_node.x, next_node.y)] = next_node

        if not goal_reached:
            print("No path found!")
        end_time = time.perf_counter()
        time_dict["BFS"] = end_time - start_time

    def visualize(self):
        start_time = time.perf_counter()
        path = []
        # Backtracking using the parent in node
        g = self.nodes_dict.pop((self.end_pt.x, self.end_pt.y))
        for _ in iter(int, 1):
            path.append(g)
            if g.parent.x == self.start_pt.x and g.parent.y == self.start_pt.y:
                break
            g = self.nodes_dict.pop((g.parent.x, g.parent.y))
        path.append(self.nodes_dict.pop((g.parent.x, g.parent.y)))
        path.reverse()
        end_time = time.perf_counter()
        time_dict["Backtracking"] = end_time - start_time

        # Creating circle to mark start and goal point in canvas
        cv2.circle(self.canvas, (path[0].x, path[0].y), 5, color=red, thickness=-1)
        cv2.circle(self.canvas, (path[-1].x, path[-1].y), 5, color=green, thickness=-1)

        video_output = cv2.VideoWriter('bfs.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps=24,
                                       frameSize=(self.canvas.shape[1], self.canvas.shape[0]))

        frame_rate = 2000
        count = 0
        # setting all visited nodes grey
        for node in self.nodes_dict:
            count += 1
            self.canvas[self.nodes_dict[node].y, self.nodes_dict[node].x] = grey
            if count == frame_rate:
                cv2.imshow("canvas", self.canvas)
                video_output.write(self.canvas)
                cv2.waitKey(1)
                count = 0

        frame_rate = 5
        count = 0

        for index in range(len(path) - 1):
            cv2.line(self.canvas, (path[index].x, path[index].y), (path[index + 1].x, path[index + 1].y), (255, 0, 0),
                     2)  # path tracing
            count += 1
            if count == frame_rate:
                cv2.imshow("canvas", self.canvas)
                video_output.write(self.canvas)
                cv2.waitKey(1)
                count = 0


        video_output.release()
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    c = Canvas(180, 50, 2, MULTIPLIER)
    # c._visualize_canvas()
    algo = str(input("BFS or DIJKSTRA "))
    if len(algo) == 0:
        print("Invalid input")
    print("Graph search using " + algo)
    GraphSearch(c.canvas, algo).visualize()
    for time in time_dict:
        print(time, time_dict[time])