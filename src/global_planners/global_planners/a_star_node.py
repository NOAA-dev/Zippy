#!/usr/bin/env python3
import rclpy
import numpy as np
import math
import cv2
import heapq
import time
from tf2_ros import Buffer, TransformListener, TransformException
from rclpy.duration import Duration
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from scipy.ndimage import distance_transform_edt
from rclpy.qos import QoSProfile, DurabilityPolicy


class node_state:
    def __init__(self, x, y, theta, v=0.0, w=0.0, g=0.0, parent=None, state=1):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v
        self.w = w
        self.g = g
        self.parent = parent
        self.state = state  # 1 for forward, -1 for backward


class AStarNode(Node):
    def __init__(self):
        super().__init__("a_star")

        path_qos = QoSProfile(depth=1)
        path_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.computing = False

        self.declare_parameter("wheelseperation", 0.5)
        self.declare_parameter("wheelradius", 0.1)
        self.declare_parameter("footprintlength", 0.6)
        self.declare_parameter("footprintwidth", 0.45)
        self.declare_parameter("footprintmargin", 0.075)

        # bot parameters
        self.L_ = self.get_parameter("wheelseperation").value
        self.R_ = self.get_parameter("wheelradius").value
        self.footprint_length = self.get_parameter("footprintlength").value
        self.footprint_width = self.get_parameter("footprintwidth").value
        self.footprint_margin = self.get_parameter("footprintmargin").value

        # system parameters
        self.dt_ = 0.5

        self.velocity_primitives_ = [
            0.8,
            0.0,
            -0.5
        ]
        self.angular_vel_primitives = np.linspace(-np.deg2rad(60), np.deg2rad(60), 5).tolist()

        ###############################
        self.x_pos = 0.0
        self.y_pos = 0.0
        self.yaw = 0.0
        self.vel_ = 0.0
        self.ang_vel_ = 0.0

        self.Grid_ = None
        self.h_map_ = None
        self.clearance_map = None
        self.map_resolution_ = None
        self.map_origin_x_ = None
        self.map_origin_y_ = None

        ##############################
        self.open_set_ = {}
        self.closed_set_ = set()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.odom_sub = self.create_subscription(Odometry, "/odometry/filtered", self.odom_callback, 10)

        self.prev_goal = None
        self.Goal_ = None
        self.goal_sub = self.create_subscription(PoseStamped, "/goal_pose", self.goal_callback, 10)

        self.tf_timer = self.create_timer(0.01, self.check_tf)

        self.path_pub = self.create_publisher(Path, "/path", path_qos)

        ###########################


        # Load PGM map
        map_path = "/home/chirag/zippy/src/program_bringup/maps/small_house/map.pgm"

        pgm_map = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)

        if pgm_map is None:
            raise RuntimeError(f"Could not load map: {map_path}")

        # Convert Nav2/ROS PGM map into planner occupancy map
        # PGM convention:
        #   254 = free
        #     0 = occupied
        #   ~205 = unknown
        #
        # Planner convention:
        #     0 = free
        #   100 = obstacle

        self.Grid_ = np.zeros_like(pgm_map, dtype=np.uint8)

        # Treat anything that is not clearly free as an obstacle
        self.Grid_[pgm_map < 250] = 100

        # Obstacle inflation
        obstacle_mask = (self.Grid_ == 100).astype(np.uint8)

        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(
            obstacle_mask,
            kernel,
            iterations=1
        )

        self.Grid_[dilated_mask == 1] = 100

        # Clearance map
        self.clearance_map = distance_transform_edt(
            self.Grid_ == 0
        )

        # EXACT values from map.yaml
        self.map_resolution_ = 0.05
        self.map_origin_x_ = -12.5
        self.map_origin_y_ = -12.5

        self.get_logger().info(f"Map loaded: {self.Grid_.shape}, "f"resolution: {self.map_resolution_}, "f"origin: ({self.map_origin_x_}, {self.map_origin_y_})")

        self._footprint_local_pts = self._build_footprint_points(self.footprint_length, self.footprint_width, self.footprint_margin, self.map_resolution_)

    def _build_footprint_points(self, length, width, margin, resolution):
        half_l = length / 2.0 + margin
        half_w = width / 2.0 + margin
        step = max(resolution / 2.0, 0.01)

        corners = [(half_l, half_w), (half_l, -half_w), (-half_l, -half_w), (-half_l, half_w)]
        pts = []
        for i in range(4):
            x0, y0 = corners[i]
            x1, y1 = corners[(i + 1) % 4]
            seg_len = math.hypot(x1 - x0, y1 - y0)
            n = max(int(seg_len / step), 1)
            for k in range(n):
                t = k / n
                pts.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))

        return np.array(pts, dtype=np.float64)

    def _footprint_grid_cells(self, x, y, theta):
        """Rotate/translate the footprint sample points to the given pose and
        return their grid indices, plus a bounds mask."""
        c, s = np.cos(theta), np.sin(theta)
        local = self._footprint_local_pts
        wx = x + local[:, 0] * c - local[:, 1] * s
        wy = y + local[:, 0] * s + local[:, 1] * c

        gx = ((wx - self.map_origin_x_) / self.map_resolution_).astype(np.int64)
        gy = ((wy - self.map_origin_y_) / self.map_resolution_).astype(np.int64)

        in_bounds = (gx >= 0) & (gy >= 0) & (gx < self.Grid_.shape[1]) & (gy < self.Grid_.shape[0])
        return gx, gy, in_bounds

    def check_tf(self):
        try:
            transform = self.tf_buffer.lookup_transform("map", "zippy", rclpy.time.Time(), timeout=Duration(seconds=0.2))

            self.x_pos = transform.transform.translation.x
            self.y_pos = transform.transform.translation.y
            z = transform.transform.translation.z
            q = transform.transform.rotation
            _, _, self.yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

        except TransformException as ex:
            pass

    def dijkstra_heuristic(self, goal_x, goal_y):
        h_map = np.full(self.Grid_.shape, np.inf, dtype=np.float32)

        pq = []
        visited_set = set()
        heapq.heappush(pq, (0.0, goal_x, goal_y))
        h_map[goal_y, goal_x] = 0.0
        motions = [(1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1),(1, 1, np.sqrt(2)), (-1, -1, np.sqrt(2)), (1, -1, np.sqrt(2)), (-1, 1, np.sqrt(2))]
        pos_x, pos_y = self.world_to_grid(self.x_pos, self.y_pos)

        while pq:
            cost, x, y = heapq.heappop(pq)
            if (x, y) in visited_set:
                continue
            visited_set.add((x, y))

            if (x, y) == (pos_x, pos_y):
                break

            for dx, dy, move_cost in motions:
                nx, ny = x + dx, y + dy
                if 0 > nx or 0 > ny or nx >= self.Grid_.shape[1] or ny >= self.Grid_.shape[0]:
                    continue
                if self.Grid_[ny, nx] != 0:
                    continue
                new_cost = cost + move_cost
                if new_cost < h_map[ny, nx]:
                    h_map[ny, nx] = new_cost
                    heapq.heappush(pq, (new_cost, nx, ny))

        return h_map

    def heuristic_cost(self, x, y):
        gx, gy = self.world_to_grid(x, y)
        if gx < 0 or gy < 0 or gx >= self.Grid_.shape[1] or gy >= self.Grid_.shape[0]:
            return np.inf
        return self.h_map_[gy, gx]

    def euler_plus_heading(self, x, y, theta):
        cost = np.hypot((self.Goal_[0] - x), (self.Goal_[1] - y))
        cost = cost + np.arctan2(np.sin(self.Goal_[2] - theta), np.cos(self.Goal_[2] - theta))
        return cost

    def discretize(self, x_grid, y_grid, theta):
        x_idx = int((x_grid - self.map_origin_x_) / self.map_resolution_)
        y_idx = int((y_grid - self.map_origin_y_) / self.map_resolution_)
        theta_bins = 36
        theta_normalized = (theta + np.pi) % (2 * np.pi)

        theta_disc = int(theta_normalized / (2 * np.pi) * theta_bins)
        return x_idx, y_idx, theta_disc

    def odom_callback(self, msg: Odometry):
        self.vel_ = msg.twist.twist.linear.x
        self.ang_vel_ = msg.twist.twist.angular.z

    def world_to_grid(self, x, y):
        gx = int((x - self.map_origin_x_) / self.map_resolution_)
        gy = int((y - self.map_origin_y_) / self.map_resolution_)
        return gx, gy

    def grid_to_world(self, gx, gy):
        x = gx * self.map_resolution_ + self.map_origin_x_
        y = gy * self.map_resolution_ + self.map_origin_y_
        return x, y

    def kinematic_model(self, x, y, theta, v, w):
        if abs(w * self.dt_) <= 0.05:
            x_new = x + v * np.cos(theta) * self.dt_
            y_new = y + v * np.sin(theta) * self.dt_
            theta_new = theta + w * self.dt_
        else:
            x_new = x + (v / w) * (np.sin(theta + w * self.dt_) - np.sin(theta))
            y_new = y + (v / w) * (np.cos(theta + w * self.dt_) - np.cos(theta))
            theta_new = theta + w * self.dt_

        return x_new, y_new, theta_new

    def collision_check(self, x, y, theta):
        gx0, gy0 = self.world_to_grid(x, y)
        if gx0 < 0 or gy0 < 0 or gx0 >= self.Grid_.shape[1] or gy0 >= self.Grid_.shape[0]:
            return True
        if self.Grid_[gy0, gx0] != 0:
            return True

        gx, gy, in_bounds = self._footprint_grid_cells(x, y, theta)
        if not np.all(in_bounds):
            return True

        if np.any(self.Grid_[gy[in_bounds], gx[in_bounds]] != 0):
            return True

        return False

    def clearance_cost(self, x, y, theta):
        gx0, gy0 = self.world_to_grid(x, y)
        if gx0 < 0 or gy0 < 0 or gx0 >= self.Grid_.shape[1] or gy0 >= self.Grid_.shape[0]:
            return np.inf
        if self.Grid_[gy0, gx0] != 0:
            return np.inf
        
        gx, gy, in_bounds = self._footprint_grid_cells(x, y, theta)
        if not np.any(in_bounds):
            return np.inf

        clearance = self.clearance_map[gy[in_bounds], gx[in_bounds]]
        min_clearance = float(np.min(clearance)) if clearance.size else 0.0

        if min_clearance <= 0.0:
            return np.inf

        return 4.0 / min_clearance

    def A_star(self, start, goal):
        self.computing = True
        self.open_set_ = []
        self.closed_set_ = set()

        start_node = node_state(start[0], start[1], start[2], 0, 0, 0, None, 1)

        heapq.heappush(self.open_set_, (0, id(start_node), start_node))

        best_cost = {}
        best_cost[self.discretize(start_node.x, start_node.y, start_node.theta)] = 0

        expansions = 0

        goal_x_idx, goal_y_idx, goal_theta_bin = self.discretize(goal[0], goal[1], goal[2])

        while self.open_set_:
            _, _, current = heapq.heappop(self.open_set_)
            current_key = self.discretize(current.x, current.y, current.theta)
            if current_key in self.closed_set_:
                continue

            self.closed_set_.add(current_key)
            expansions += 1
            if expansions >= 1000:
                break

            dist_to_goal = math.hypot(current.x - goal[0], current.y - goal[1])
            heading_err = abs(math.atan2(math.sin(current.theta - goal[2]), math.cos(current.theta - goal[2])))

            if dist_to_goal < 0.15 and heading_err < math.radians(20):
                self.get_logger().info("Goal reached!")
                self.get_logger().info(f"Expansions: {expansions}, Open set size: {len(self.open_set_)}, Closed set size: {len(self.closed_set_)}")

                path = []
                node = current
                while node is not None:
                    path.append((node.x, node.y, node.theta))
                    node = node.parent

                return path[::-1]

            for w in self.angular_vel_primitives:
                for v in self.velocity_primitives_:

                    x_new, y_new, theta_new = self.kinematic_model(current.x, current.y, current.theta, v, w)
                    theta_new = np.arctan2(np.sin(theta_new), np.cos(theta_new))

                    collision = self.collision_check(x_new, y_new, theta_new)
                    if collision == True:
                        continue

                    if v < 0.0:
                        reverse_cost = 10
                        state = -1
                    else:
                        reverse_cost = 0
                        state = 1

                    c = self.clearance_cost(x_new, y_new, theta_new)
                    h_cost = max(self.heuristic_cost(x_new, y_new), self.euler_plus_heading(x_new, y_new, theta_new))
                    g_cost = current.g + self.dt_ * (abs(v) + abs(w)) + reverse_cost + c

                    new_node = node_state(x_new, y_new, theta_new, v, w, g_cost, current, state)
                    new_key = self.discretize(x_new, y_new, theta_new)

                    if new_key not in best_cost or g_cost < best_cost[new_key]:
                        best_cost[new_key] = g_cost
                        f_cost = h_cost + g_cost
                        heapq.heappush(self.open_set_, (f_cost, id(new_node), new_node))

        self.get_logger().warn("No path found!")
        return None

    def goal_callback(self, msg: PoseStamped):
        if self.Grid_ is None:
            self.get_logger().warn("Map not received yet!")
            return

        goal_x = msg.pose.position.x
        goal_y = msg.pose.position.y
        q = msg.pose.orientation
        _, _, goal_theta = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.Goal_ = (goal_x, goal_y, goal_theta)

        if self.prev_goal is not None and self.prev_goal == self.Goal_:
            return

        t0 = time.perf_counter()

        self.h_map_ = self.dijkstra_heuristic(*self.world_to_grid(goal_x, goal_y))

        start = (self.x_pos, self.y_pos, self.yaw)
        goal = (goal_x, goal_y, goal_theta)

        path = self.A_star(start, goal)

        elapsed = time.perf_counter() - t0
        self.get_logger().info(f"Path planning time: {elapsed:.6f} seconds")
        self.computing = False
        if path is not None:
            path_msg = Path()
            path_msg.header.frame_id = "map"
            path_msg.header.stamp = self.get_clock().now().to_msg()
            for x, y, theta in path:
                pose_stamped = PoseStamped()
                pose_stamped.header.frame_id = "map"
                pose_stamped.pose.position.x = x
                pose_stamped.pose.position.y = y
                q = quaternion_from_euler(0, 0, theta)
                pose_stamped.pose.orientation.x = q[0]
                pose_stamped.pose.orientation.y = q[1]
                pose_stamped.pose.orientation.z = q[2]
                pose_stamped.pose.orientation.w = q[3]
                path_msg.poses.append(pose_stamped)

            self.path_pub.publish(path_msg)
            self.prev_goal = self.Goal_


def main(args=None):
    rclpy.init(args=args)
    node = AStarNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()