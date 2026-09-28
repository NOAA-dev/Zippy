#!/usr/bin/env python3
# import rclpy
# import numpy as np
# import math
# import cv2
# import heapq
# import time
# from skimage.graph import MCP_Geometric
# from tf2_ros import Buffer, TransformListener, TransformException
# from rclpy.duration import Duration
# from nav_msgs.msg import Path
# from geometry_msgs.msg import PoseStamped
# from rclpy.node import Node
# from nav_msgs.msg import Odometry
# from tf_transformations import euler_from_quaternion, quaternion_from_euler
# from scipy.ndimage import distance_transform_edt
# from rclpy.qos import QoSProfile, DurabilityPolicy


# class node_state:
#     def __init__(self, x, y, theta, v=0.0, w=0.0, g=0.0, parent=None, state=1):
#         self.x = x
#         self.y = y
#         self.theta = theta
#         self.v = v
#         self.w = w
#         self.g = g
#         self.parent = parent
#         self.state = state  # 1 for forward, -1 for backward


# class AStarNode(Node):
#     def __init__(self):
#         super().__init__("a_star")

#         path_qos = QoSProfile(depth=1)
#         path_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
#         self.computing = False

#         self.declare_parameter("wheelseperation", 0.51)
#         self.declare_parameter("wheelradius", 0.1)
#         self.declare_parameter("footprintlength", 0.6)
#         self.declare_parameter("footprintwidth", 0.45)
#         self.declare_parameter("footprintmargin", 0.05)
#         self.declare_parameter("search_resolution", 0.4)
#         self.declare_parameter("search_theta_bins", 20)

#         self.declare_parameter("heading_cost_weight", 0.5)
#         self.declare_parameter("clearance_cost_weight", 0.2)
#         self.declare_parameter("reverse_move_penalty", 10)
#         self.declare_parameter("smoothing_enabled", True)
#         self.declare_parameter("smoothing_corner_iterations", 2)
#         self.declare_parameter("smoothing_resample_step", 0.1)   # metres between output poses


#         # bot parameters
#         self.L_ = self.get_parameter("wheelseperation").value
#         self.R_ = self.get_parameter("wheelradius").value
#         self.footprint_length = self.get_parameter("footprintlength").value
#         self.footprint_width = self.get_parameter("footprintwidth").value
#         self.footprint_margin = self.get_parameter("footprintmargin").value


#         self.search_resolution_ = self.get_parameter("search_resolution").value
#         self.search_theta_bins_ = self.get_parameter("search_theta_bins").value
#         self.heading_cost_weight = self.get_parameter("heading_cost_weight").value
#         self.clearance_cost_weight = self.get_parameter("clearance_cost_weight").value
#         self.reverse_move_penalty = self.get_parameter("reverse_move_penalty").value

#         self.smoothing_enabled = self.get_parameter("smoothing_enabled").value
#         self.smoothing_corner_iterations = self.get_parameter("smoothing_corner_iterations").value
#         self.smoothing_resample_step = self.get_parameter("smoothing_resample_step").value

#         # system parameters
#         self.dt_ = 0.5

#         self.velocity_primitives_ = [
#             0.8,
#             0.0,
#             -0.5
#         ]

#         self.angular_vel_primitives = np.linspace(-np.deg2rad(60), np.deg2rad(60), 5).tolist()

#         ###############################
#         self.x_pos = 0.0
#         self.y_pos = 0.0
#         self.yaw = 0.0
#         self.vel_ = 0.0
#         self.ang_vel_ = 0.0

#         self.Grid_ = None
#         self.h_map_ = None
#         self.clearance_map = None
#         self.map_resolution_ = None
#         self.map_origin_x_ = None
#         self.map_origin_y_ = None

#         ##############################
#         self.open_set_ = {}
#         self.closed_set_ = set()

#         self.tf_buffer = Buffer()
#         self.tf_listener = TransformListener(self.tf_buffer, self)
#         self.odom_sub = self.create_subscription(Odometry, "/odometry/filtered", self.odom_callback, 10)

#         self.prev_goal = None
#         self.Goal_ = None
#         self.goal_sub = self.create_subscription(PoseStamped, "/goal_pose", self.goal_callback, 10)

#         self.tf_timer = self.create_timer(0.01, self.check_tf)

#         self.path_pub = self.create_publisher(Path, "/path", path_qos)

#         ###########################


#         # Load PGM map
#         map_path = "/home/chirag/zippy/src/program_bringup/maps/object_avoider/map.pgm"

#         pgm_map = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)

#         if pgm_map is None:
#             raise RuntimeError(f"Could not load map: {map_path}")
#         pgm_map = cv2.flip(pgm_map, 0)

#         self.Grid_ = np.zeros_like(pgm_map, dtype=np.int8)
#         # 0 = free, 100 = known obstacle, -1 = unknown
#         self.Grid_[(pgm_map >= 10) & (pgm_map < 250)] = -1
#         self.Grid_[pgm_map < 10] = 100
#         # Obstacle inflation (only known obstacles get inflated, not unknown space)
#         obstacle_mask = (self.Grid_ == 100).astype(np.uint8)
#         kernel = np.ones((3, 3), np.uint8)
#         dilated_mask = cv2.dilate(obstacle_mask,kernel,iterations=1)
#         self.Grid_[dilated_mask == 1] = 100
#         # Clearance map
#         self.clearance_map = distance_transform_edt(self.Grid_ == 0)

#         # EXACT values from map.yaml
#         self.map_resolution_ = 0.05
#         self.map_origin_x_ = -10.277
#         self.map_origin_y_ = -10.001

#         self.get_logger().info(f"Map loaded: {self.Grid_.shape}, "f"resolution: {self.map_resolution_}, "f"origin: ({self.map_origin_x_}, {self.map_origin_y_})")

#         self._footprint_local_pts = self._build_footprint_points(self.footprint_length, self.footprint_width, self.footprint_margin, self.map_resolution_)

#     def _build_footprint_points(self, length, width, margin, resolution):
#         half_l = length / 2.0 + margin
#         half_w = width / 2.0 + margin
#         step = max(resolution / 2.0, 0.01)

#         corners = [(half_l, half_w), (half_l, -half_w), (-half_l, -half_w), (-half_l, half_w)]
#         pts = []
#         for i in range(4):
#             x0, y0 = corners[i]
#             x1, y1 = corners[(i + 1) % 4]
#             seg_len = math.hypot(x1 - x0, y1 - y0)
#             n = max(int(seg_len / step), 1)
#             for k in range(n):
#                 t = k / n
#                 pts.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))

#         return np.array(pts, dtype=np.float64)

#     def _footprint_grid_cells(self, x, y, theta):
#         c, s = np.cos(theta), np.sin(theta)
#         local = self._footprint_local_pts
#         wx = x + local[:, 0] * c - local[:, 1] * s
#         wy = y + local[:, 0] * s + local[:, 1] * c

#         gx = ((wx - self.map_origin_x_) / self.map_resolution_).astype(np.int64)
#         gy = ((wy - self.map_origin_y_) / self.map_resolution_).astype(np.int64)

#         in_bounds = (gx >= 0) & (gy >= 0) & (gx < self.Grid_.shape[1]) & (gy < self.Grid_.shape[0])
#         return gx, gy, in_bounds

#     def check_tf(self):
#         try:
#             transform = self.tf_buffer.lookup_transform("map", "zippy", rclpy.time.Time(), timeout=Duration(seconds=0.2))

#             self.x_pos = transform.transform.translation.x
#             self.y_pos = transform.transform.translation.y
#             z = transform.transform.translation.z
#             q = transform.transform.rotation
#             _, _, self.yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

#         except TransformException as ex:
#             pass

#     def dijkstra_heuristic(self, goal_x, goal_y):
#         cost_array = np.where(self.Grid_ == 0, 1.0, np.inf).astype(np.float64)
#         mcp = MCP_Geometric(cost_array, fully_connected=True)
#         costs, _ = mcp.find_costs([(goal_y, goal_x)])
#         return (costs).astype(np.float32)

#     def heuristic_cost(self, x, y):
#         gx, gy = self.world_to_grid(x, y)
#         if gx < 0 or gy < 0 or gx >= self.Grid_.shape[1] or gy >= self.Grid_.shape[0]:
#             return np.inf
#         return self.h_map_[gy, gx]

#     def euler_plus_heading(self, x, y, theta):
#         cost = np.hypot((self.Goal_[0] - x), (self.Goal_[1] - y))
#         heading_err = abs(np.arctan2(np.sin(self.Goal_[2] - theta), np.cos(self.Goal_[2] - theta)))
#         cost = cost + self.heading_cost_weight * heading_err
#         return cost/self.map_resolution_

#     def discretize(self, x_grid, y_grid, theta):
#         x_idx = int((x_grid - self.map_origin_x_) / self.search_resolution_)
#         y_idx = int((y_grid - self.map_origin_y_) / self.search_resolution_)
#         theta_bins = self.search_theta_bins_
#         theta_normalized = (theta + np.pi) % (2 * np.pi)

#         theta_disc = int(theta_normalized / (2 * np.pi) * theta_bins)
#         return x_idx, y_idx, theta_disc

#     def odom_callback(self, msg: Odometry):
#         self.vel_ = msg.twist.twist.linear.x
#         self.ang_vel_ = msg.twist.twist.angular.z

#     def world_to_grid(self, x, y):
#         gx = int((x - self.map_origin_x_) / self.map_resolution_)
#         gy = int((y - self.map_origin_y_) / self.map_resolution_)
#         return gx, gy

#     def grid_to_world(self, gx, gy):
#         x = gx * self.map_resolution_ + self.map_origin_x_
#         y = gy * self.map_resolution_ + self.map_origin_y_
#         return x, y

#     def kinematic_model(self, x, y, theta, v, w):
#         if abs(w * self.dt_) <= 0.05:
#             x_new = x + v * np.cos(theta) * self.dt_
#             y_new = y + v * np.sin(theta) * self.dt_
#             theta_new = theta + w * self.dt_
#         else:
#             x_new = x + (v / w) * (np.sin(theta + w * self.dt_) - np.sin(theta))
#             y_new = y + (v / w) * (np.cos(theta + w * self.dt_) - np.cos(theta))
#             theta_new = theta + w * self.dt_

#         return x_new, y_new, theta_new

#     def _pose_footprint_status(self, x, y, theta):
#         gx0, gy0 = self.world_to_grid(x, y)
#         if gx0 < 0 or gy0 < 0 or gx0 >= self.Grid_.shape[1] or gy0 >= self.Grid_.shape[0]:
#             return True, np.inf
#         if self.Grid_[gy0, gx0] != 0:
#             return True, np.inf

#         gx, gy, in_bounds = self._footprint_grid_cells(x, y, theta)
#         if not np.all(in_bounds):
#             return True, np.inf

#         if np.any(self.Grid_[gy[in_bounds], gx[in_bounds]] != 0):
#             return True, np.inf

#         clearance = self.clearance_map[gy[in_bounds], gx[in_bounds]] * self.map_resolution_
#         min_clearance = float(np.min(clearance)) if clearance.size else 0.0

#         if min_clearance <= 0.0:
#             return False, np.inf

#         return False, self.clearance_cost_weight / min_clearance

#     def collision_check(self, x, y, theta):
#         collision, _ = self._pose_footprint_status(x, y, theta)
#         return collision

#     def clearance_cost(self, x, y, theta):
#         _, clearance = self._pose_footprint_status(x, y, theta)
#         return clearance

#     # ------------------------------------------------------------------
#     # Path smoothing
#     # ------------------------------------------------------------------
#     def _segment_heading(self, p0, p1, state):
#         h = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
#         if state < 0:                       # reversing: robot faces opposite to motion
#             h += math.pi
#         return math.atan2(math.sin(h), math.cos(h))

#     def _segment_free(self, p0, p1, state):
#         """True if driving straight from p0 to p1 keeps the whole footprint collision-free."""
#         dist = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
#         if dist < 1e-6:
#             return True
#         heading = self._segment_heading(p0, p1, state)
#         step = self.map_resolution_ / 2.0
#         n = max(int(dist / step), 1)
#         for k in range(n + 1):
#             t = k / n
#             x = p0[0] + (p1[0] - p0[0]) * t
#             y = p0[1] + (p1[1] - p0[1]) * t
#             if self.collision_check(x, y, heading):
#                 return False
#         return True

#     def _shortcut(self, pts, state):
#         """Greedy line-of-sight shortcutting: from each kept point jump to the farthest visible one."""
#         out = [pts[0]]
#         i = 0
#         while i < len(pts) - 1:
#             j = len(pts) - 1
#             while j > i + 1 and not self._segment_free(pts[i], pts[j], state):
#                 j -= 1
#             out.append(pts[j])
#             i = j
#         return out

#     def _round_corners(self, pts, state, iterations):
#         """Chaikin corner cutting; a corner is only replaced if the new segments are collision-free."""
#         for _ in range(iterations):
#             if len(pts) < 3:
#                 break
#             new_pts = [pts[0]]
#             for k in range(1, len(pts) - 1):
#                 p_prev, p, p_next = pts[k - 1], pts[k], pts[k + 1]
#                 q = (0.75 * p[0] + 0.25 * p_next[0], 0.75 * p[1] + 0.25 * p_next[1])
#                 r_prev = (0.75 * p[0] + 0.25 * p_prev[0], 0.75 * p[1] + 0.25 * p_prev[1])
#                 # cut the corner: p_prev-side point -> next-side point
#                 if (self._segment_free(new_pts[-1], r_prev, state)
#                         and self._segment_free(r_prev, q, state)
#                         and self._segment_free(q, p_next, state)):
#                     new_pts.extend([r_prev, q])
#                 else:
#                     new_pts.append(p)
#             new_pts.append(pts[-1])
#             pts = new_pts
#         return pts

#     def _resample(self, pts, step):
#         out = [pts[0]]
#         for a, b in zip(pts[:-1], pts[1:]):
#             d = math.hypot(b[0] - a[0], b[1] - a[1])
#             n = max(int(math.ceil(d / step)), 1)
#             for k in range(1, n + 1):
#                 t = k / n
#                 out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
#         return out

#     def smooth_path(self, path):
#         """path: list of (x, y, theta, state). Returns list of (x, y, theta)."""
#         if len(path) < 3:
#             return [(x, y, th) for x, y, th, _ in path]

#         # split into runs of constant direction (segment i-1 -> i has direction path[i].state)
#         runs = []
#         cur = [path[0][:2]]
#         cur_state = path[1][3]
#         for i in range(1, len(path)):
#             s = path[i][3]
#             if s != cur_state:
#                 runs.append((cur_state, cur))
#                 cur = [path[i - 1][:2]]
#                 cur_state = s
#             cur.append(path[i][:2])
#         runs.append((cur_state, cur))

#         result = []
#         for idx, (state, pts) in enumerate(runs):
#             pts = self._shortcut(pts, state)
#             pts = self._round_corners(pts, state, self.smoothing_corner_iterations)
#             pts = self._resample(pts, self.smoothing_resample_step)

#             for k, p in enumerate(pts):
#                 if idx > 0 and k == 0:
#                     continue                      # cusp point already added by previous run
#                 nxt = pts[k + 1] if k + 1 < len(pts) else None
#                 if nxt is not None:
#                     theta = self._segment_heading(p, nxt, state)
#                 else:
#                     theta = result[-1][2] if result else path[0][2]
#                 result.append((p[0], p[1], theta))

#         # keep the true start and goal orientations
#         result[0] = (result[0][0], result[0][1], path[0][2])
#         result[-1] = (result[-1][0], result[-1][1], path[-1][2])
#         return result

#     def A_star(self, start, goal):
#         self.computing = True
#         self.open_set_ = []
#         self.closed_set_ = set()

#         start_node = node_state(start[0], start[1], start[2], 0, 0, 0, None, 1)

#         heapq.heappush(self.open_set_, (0, id(start_node), start_node))

#         best_cost = {}
#         best_cost[self.discretize(start_node.x, start_node.y, start_node.theta)] = 0

#         expansions = 0

#         goal_x_idx, goal_y_idx, goal_theta_bin = self.discretize(goal[0], goal[1], goal[2])

#         while self.open_set_:
#             _, _, current = heapq.heappop(self.open_set_)
#             current_key = self.discretize(current.x, current.y, current.theta)
#             if current_key in self.closed_set_:
#                 continue

#             self.closed_set_.add(current_key)
#             expansions += 1
#             if expansions >= 10000:
#                 break

#             dist_to_goal = math.hypot(current.x - goal[0], current.y - goal[1])
#             heading_err = abs(math.atan2(math.sin(current.theta - goal[2]), math.cos(current.theta - goal[2])))

#             if dist_to_goal < 0.4 and heading_err < math.radians(25):
#                 self.get_logger().info("Goal reached!")
#                 self.get_logger().info(f"Expansions: {expansions}, Open set size: {len(self.open_set_)}, Closed set size: {len(self.closed_set_)}")

#                 path = []
#                 node = current
#                 while node is not None:
#                     path.append((node.x, node.y, node.theta, node.state))
#                     node = node.parent

#                 return path[::-1]

#             for w in self.angular_vel_primitives:
#                 for v in self.velocity_primitives_:

#                     x_new, y_new, theta_new = self.kinematic_model(current.x, current.y, current.theta, v, w)
#                     theta_new = np.arctan2(np.sin(theta_new), np.cos(theta_new))

#                     collision, c = self._pose_footprint_status(x_new, y_new, theta_new)
#                     if collision:
#                         continue

#                     if v < 0.0:
#                         reverse_cost = self.reverse_move_penalty
#                         state = -1
#                     else:
#                         reverse_cost = 0
#                         state = 1
#                     if dist_to_goal >= 1.0:
#                         h_cost = self.heuristic_cost(x_new, y_new)
#                     else:
#                         h_cost = self.euler_plus_heading(x_new, y_new, theta_new)
#                     g_cost = current.g + self.dt_ * abs(v) + self.dt_ * abs(w) + reverse_cost + c

#                     new_node = node_state(x_new, y_new, theta_new, v, w, g_cost, current, state)
#                     new_key = self.discretize(x_new, y_new, theta_new)

#                     if new_key not in best_cost or g_cost < best_cost[new_key]:
#                         best_cost[new_key] = g_cost
#                         f_cost = h_cost + g_cost
#                         heapq.heappush(self.open_set_, (f_cost, id(new_node), new_node))

#         self.get_logger().warn("No path found!")
#         return None

#     def goal_callback(self, msg: PoseStamped):
#         if self.Grid_ is None:
#             self.get_logger().warn("Map not received yet!")
#             return

#         goal_x = msg.pose.position.x
#         goal_y = msg.pose.position.y
#         q = msg.pose.orientation
#         _, _, goal_theta = euler_from_quaternion([q.x, q.y, q.z, q.w])
#         self.Goal_ = (goal_x, goal_y, goal_theta)

#         if self.prev_goal is not None and self.prev_goal == self.Goal_:
#             return

#         t0 = time.perf_counter()

#         self.h_map_ = self.dijkstra_heuristic(*self.world_to_grid(goal_x, goal_y))

#         start = (self.x_pos, self.y_pos, self.yaw)
#         goal = (goal_x, goal_y, goal_theta)

#         path = self.A_star(start, goal)

#         elapsed = time.perf_counter() - t0
#         self.get_logger().info(f"Path planning time: {elapsed:.6f} seconds")
#         self.computing = False

#         if path is not None:
#             n_raw = len(path)
#             if self.smoothing_enabled:
#                 t1 = time.perf_counter()
#                 path = self.smooth_path(path)
#                 self.get_logger().info(
#                     f"Smoothing: {n_raw} raw poses -> {len(path)} poses in {time.perf_counter() - t1:.4f} s")
#             else:
#                 path = [(x, y, th) for x, y, th, _ in path]

#         if path is not None:
#             path_msg = Path()
#             path_msg.header.frame_id = "map"
#             path_msg.header.stamp = self.get_clock().now().to_msg()
#             for x, y, theta in path:
#                 pose_stamped = PoseStamped()
#                 pose_stamped.header.frame_id = "map"
#                 pose_stamped.pose.position.x = x
#                 pose_stamped.pose.position.y = y
#                 q = quaternion_from_euler(0, 0, theta)
#                 pose_stamped.pose.orientation.x = q[0]
#                 pose_stamped.pose.orientation.y = q[1]
#                 pose_stamped.pose.orientation.z = q[2]
#                 pose_stamped.pose.orientation.w = q[3]
#                 path_msg.poses.append(pose_stamped)

#             self.path_pub.publish(path_msg)
#             self.prev_goal = self.Goal_


# def main(args=None):
#     rclpy.init(args=args)
#     node = AStarNode()
#     rclpy.spin(node)
#     rclpy.shutdown()


# if __name__ == "__main__":
#     main()


import rclpy
import numpy as np
import math
import cv2
import time
from skimage.graph import MCP_Geometric
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
    __slots__ = ("x", "y", "theta", "v", "w", "g", "parent", "state")

    def __init__(self, x, y, theta, v=0.0, w=0.0, g=0.0, parent=None, state=1):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v
        self.w = w
        self.g = g
        self.parent = parent
        self.state = state  # 1 for forward, -1 for backward


class _BucketQueue:
    """Approximate priority queue. O(1) push; amortized O(1) pop, since A*
    costs are non-decreasing along the search frontier so the scan pointer
    only ever moves forward. Within one bucket, order is arbitrary (LIFO) --
    fine as long as `resolution` is smaller than any cost difference you
    actually care about. This trades exact best-first ordering for speed:
    path cost is only guaranteed optimal to within `resolution`.
    """

    __slots__ = ("resolution", "buckets", "min_bucket", "count")

    def __init__(self, resolution=0.05):
        self.resolution = resolution
        self.buckets = {}
        self.min_bucket = 0
        self.count = 0

    def push(self, cost, item):
        b = int(cost / self.resolution)
        bucket = self.buckets.get(b)
        if bucket is None:
            self.buckets[b] = [item]
        else:
            bucket.append(item)
        if b < self.min_bucket:
            self.min_bucket = b
        self.count += 1

    def pop(self):
        buckets = self.buckets
        b = self.min_bucket
        while not buckets.get(b):
            buckets.pop(b, None)
            b += 1
        self.min_bucket = b
        item = buckets[b].pop()
        self.count -= 1
        return item

    def __len__(self):
        return self.count

    def __bool__(self):
        return self.count > 0


class AStarNode(Node):
    def __init__(self):
        super().__init__("a_star")

        path_qos = QoSProfile(depth=1)
        path_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.computing = False

        self.declare_parameter("wheelseperation", 0.51)
        self.declare_parameter("wheelradius", 0.1)
        self.declare_parameter("footprintlength", 0.6)
        self.declare_parameter("footprintwidth", 0.45)
        self.declare_parameter("footprintmargin", 0.05)
        self.declare_parameter("search_resolution", 0.4)
        self.declare_parameter("search_theta_bins", 20)

        self.declare_parameter("heading_cost_weight", 0.5)
        self.declare_parameter("clearance_cost_weight", 0.2)
        self.declare_parameter("reverse_move_penalty", 10)

        self.declare_parameter("smoothing_enabled", True)
        self.declare_parameter("smoothing_corner_iterations", 2)
        self.declare_parameter("smoothing_resample_step", 0.1)  # metres between output poses

        self.declare_parameter("motion_lookup_theta_bins", 720)  # 0.5 deg resolution
        self.declare_parameter("open_set_bucket_resolution", 0.05)

        # bot parameters
        self.L_ = self.get_parameter("wheelseperation").value
        self.R_ = self.get_parameter("wheelradius").value
        self.footprint_length = self.get_parameter("footprintlength").value
        self.footprint_width = self.get_parameter("footprintwidth").value
        self.footprint_margin = self.get_parameter("footprintmargin").value

        self.search_resolution_ = self.get_parameter("search_resolution").value
        self.search_theta_bins_ = self.get_parameter("search_theta_bins").value
        self.heading_cost_weight = self.get_parameter("heading_cost_weight").value
        self.clearance_cost_weight = self.get_parameter("clearance_cost_weight").value
        self.reverse_move_penalty = self.get_parameter("reverse_move_penalty").value

        self.smoothing_enabled = self.get_parameter("smoothing_enabled").value
        self.smoothing_corner_iterations = self.get_parameter("smoothing_corner_iterations").value
        self.smoothing_resample_step = self.get_parameter("smoothing_resample_step").value

        self.motion_lookup_theta_bins = self.get_parameter("motion_lookup_theta_bins").value
        self.open_set_bucket_resolution = self.get_parameter("open_set_bucket_resolution").value

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
        self.open_set_ = None
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
        map_path = "/home/chirag/zippy/src/program_bringup/maps/object_avoider/map.pgm"

        pgm_map = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)

        if pgm_map is None:
            raise RuntimeError(f"Could not load map: {map_path}")
        pgm_map = cv2.flip(pgm_map, 0)

        self.Grid_ = np.zeros_like(pgm_map, dtype=np.int8)
        # 0 = free, 100 = known obstacle, -1 = unknown
        self.Grid_[(pgm_map >= 10) & (pgm_map < 250)] = -1
        self.Grid_[pgm_map < 10] = 100
        # Obstacle inflation (only known obstacles get inflated, not unknown space)
        obstacle_mask = (self.Grid_ == 100).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(obstacle_mask, kernel, iterations=1)
        self.Grid_[dilated_mask == 1] = 100
        # Clearance map
        self.clearance_map = distance_transform_edt(self.Grid_ == 0)

        # EXACT values from map.yaml
        self.map_resolution_ = 0.05
        self.map_origin_x_ = -10.277
        self.map_origin_y_ = -10.001

        self.get_logger().info(f"Map loaded: {self.Grid_.shape}, "
                                f"resolution: {self.map_resolution_}, "
                                f"origin: ({self.map_origin_x_}, {self.map_origin_y_})")

        self._footprint_local_pts = self._build_footprint_points(
            self.footprint_length, self.footprint_width, self.footprint_margin, self.map_resolution_
        )

        # circumradius: farthest footprint point from center -> beyond this clearance, footprint is
        # guaranteed free at ANY orientation.
        # inradius: largest circle fully inside the footprint rectangle -> within this clearance, an
        # obstacle is guaranteed to intersect the footprint at ANY orientation.
        half_l = self.footprint_length / 2.0 + self.footprint_margin
        half_w = self.footprint_width / 2.0 + self.footprint_margin
        self.footprint_circumradius = math.hypot(half_l, half_w)
        self.footprint_inradius = min(half_l, half_w)

        self._build_motion_primitive_table()

    # ------------------------------------------------------------------
    # Footprint / motion primitive setup
    # ------------------------------------------------------------------
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

    def _build_motion_primitive_table(self):
        """Precompute (dx, dy) for every (velocity, angular velocity) primitive at every
        discretized heading bin, so the search loop only does table lookups instead of
        repeated sin/cos evaluation."""
        N = self.motion_lookup_theta_bins
        dt = self.dt_
        two_pi = 2.0 * math.pi

        # primitive metadata that does NOT depend on theta (n_w x n_v entries)
        self._primitives = []
        for w in self.angular_vel_primitives:
            for v in self.velocity_primitives_:
                dtheta = w * dt
                if v < 0.0:
                    reverse_cost, state = self.reverse_move_penalty, -1
                else:
                    reverse_cost, state = 0, 1
                move_cost_base = dt * abs(v) + dt * abs(w) + reverse_cost
                self._primitives.append((v, w, dtheta, move_cost_base, state))

        n_prim = len(self._primitives)
        self._dx_table = [[0.0] * n_prim for _ in range(N)]
        self._dy_table = [[0.0] * n_prim for _ in range(N)]

        for b in range(N):
            theta = -math.pi + (b + 0.5) * (two_pi / N)  # bin-center heading
            for p_idx, (v, w, dtheta, _, _) in enumerate(self._primitives):
                if abs(w * dt) <= 0.05:
                    dx = v * math.cos(theta) * dt
                    dy = v * math.sin(theta) * dt
                else:
                    dx = (v / w) * (math.sin(theta + w * dt) - math.sin(theta))
                    dy = (v / w) * (math.cos(theta + w * dt) - math.cos(theta))
                self._dx_table[b][p_idx] = dx
                self._dy_table[b][p_idx] = dy

        self._n_theta_bins_motion = N
        self._theta_bin_scale = N / two_pi

    def _theta_to_motion_bin(self, theta):
        b = int((theta + math.pi) * self._theta_bin_scale)
        if b >= self._n_theta_bins_motion:
            b = self._n_theta_bins_motion - 1
        elif b < 0:
            b = 0
        return b

    @staticmethod
    def _wrap_angle(theta):
        theta = math.fmod(theta + math.pi, 2.0 * math.pi)
        if theta < 0.0:
            theta += 2.0 * math.pi
        return theta - math.pi

    def _footprint_grid_cells(self, x, y, theta):
        c, s = math.cos(theta), math.sin(theta)
        local = self._footprint_local_pts
        wx = x + local[:, 0] * c - local[:, 1] * s
        wy = y + local[:, 0] * s + local[:, 1] * c

        gx = ((wx - self.map_origin_x_) / self.map_resolution_).astype(np.int64)
        gy = ((wy - self.map_origin_y_) / self.map_resolution_).astype(np.int64)

        in_bounds = (gx >= 0) & (gy >= 0) & (gx < self.Grid_.shape[1]) & (gy < self.Grid_.shape[0])
        return gx, gy, in_bounds

    def check_tf(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                "map", "zippy", rclpy.time.Time(), timeout=Duration(seconds=0.2)
            )

            self.x_pos = transform.transform.translation.x
            self.y_pos = transform.transform.translation.y
            q = transform.transform.rotation
            _, _, self.yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

        except TransformException:
            pass

    def dijkstra_heuristic(self, goal_x, goal_y):
        cost_array = np.where(self.Grid_ == 0, 1.0, np.inf).astype(np.float64)
        mcp = MCP_Geometric(cost_array, fully_connected=True)
        costs, _ = mcp.find_costs([(goal_y, goal_x)])
        return costs.astype(np.float32)

    def heuristic_cost(self, x, y):
        gx, gy = self.world_to_grid(x, y)
        if gx < 0 or gy < 0 or gx >= self.Grid_.shape[1] or gy >= self.Grid_.shape[0]:
            return math.inf
        return float(self.h_map_[gy, gx])

    def euler_plus_heading(self, x, y, theta):
        cost = math.hypot(self.Goal_[0] - x, self.Goal_[1] - y)
        heading_err = abs(math.atan2(math.sin(self.Goal_[2] - theta), math.cos(self.Goal_[2] - theta)))
        cost = cost + self.heading_cost_weight * heading_err
        return cost / self.map_resolution_

    def discretize(self, x_grid, y_grid, theta):
        x_idx = int((x_grid - self.map_origin_x_) / self.search_resolution_)
        y_idx = int((y_grid - self.map_origin_y_) / self.search_resolution_)
        theta_bins = self.search_theta_bins_
        theta_normalized = math.fmod(theta + math.pi, 2.0 * math.pi)
        if theta_normalized < 0.0:
            theta_normalized += 2.0 * math.pi

        theta_disc = int(theta_normalized / (2.0 * math.pi) * theta_bins)
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

    # ------------------------------------------------------------------
    # Collision checking (two-tier: broad-phase circle test, then footprint)
    # ------------------------------------------------------------------
    def _pose_footprint_status(self, x, y, theta):
        gx0, gy0 = self.world_to_grid(x, y)
        if gx0 < 0 or gy0 < 0 or gx0 >= self.Grid_.shape[1] or gy0 >= self.Grid_.shape[0]:
            return True, math.inf
        if self.Grid_[gy0, gx0] != 0:
            return True, math.inf

        clearance_center = self.clearance_map[gy0, gx0] * self.map_resolution_

        # Broad phase: center is far enough from any obstacle that the whole footprint
        # is guaranteed free regardless of heading -> skip the footprint check entirely.
        if clearance_center >= self.footprint_circumradius:
            safe = clearance_center - self.footprint_circumradius
            if safe <= 0.0:
                safe = 1e-3
            return False, self.clearance_cost_weight / safe

        # Broad phase: center is close enough that an obstacle is guaranteed to be inside
        # the footprint regardless of heading -> reject immediately.
        if clearance_center <= self.footprint_inradius:
            return True, math.inf

        # Narrow phase: ambiguous band, need the real per-point footprint check.
        gx, gy, in_bounds = self._footprint_grid_cells(x, y, theta)
        if not np.all(in_bounds):
            return True, math.inf

        if np.any(self.Grid_[gy[in_bounds], gx[in_bounds]] != 0):
            return True, math.inf

        clearance = self.clearance_map[gy[in_bounds], gx[in_bounds]] * self.map_resolution_
        min_clearance = float(np.min(clearance)) if clearance.size else 0.0

        if min_clearance <= 0.0:
            return False, math.inf

        return False, self.clearance_cost_weight / min_clearance

    def collision_check(self, x, y, theta):
        collision, _ = self._pose_footprint_status(x, y, theta)
        return collision

    def clearance_cost(self, x, y, theta):
        _, clearance = self._pose_footprint_status(x, y, theta)
        return clearance

    # ------------------------------------------------------------------
    # Path smoothing
    # ------------------------------------------------------------------
    def _segment_heading(self, p0, p1, state):
        h = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        if state < 0:  # reversing: robot faces opposite to motion
            h += math.pi
        return math.atan2(math.sin(h), math.cos(h))

    def _segment_free(self, p0, p1, state):
        """True if driving straight from p0 to p1 keeps the whole footprint collision-free."""
        dist = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        if dist < 1e-6:
            return True
        heading = self._segment_heading(p0, p1, state)
        step = self.map_resolution_ / 2.0
        n = max(int(dist / step), 1)
        for k in range(n + 1):
            t = k / n
            x = p0[0] + (p1[0] - p0[0]) * t
            y = p0[1] + (p1[1] - p0[1]) * t
            if self.collision_check(x, y, heading):
                return False
        return True

    def _shortcut(self, pts, state):
        """Greedy line-of-sight shortcutting: from each kept point jump to the farthest visible one."""
        out = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self._segment_free(pts[i], pts[j], state):
                j -= 1
            out.append(pts[j])
            i = j
        return out

    def _round_corners(self, pts, state, iterations):
        """Chaikin corner cutting; a corner is only replaced if the new segments are collision-free."""
        for _ in range(iterations):
            if len(pts) < 3:
                break
            new_pts = [pts[0]]
            for k in range(1, len(pts) - 1):
                p_prev, p, p_next = pts[k - 1], pts[k], pts[k + 1]
                q = (0.75 * p[0] + 0.25 * p_next[0], 0.75 * p[1] + 0.25 * p_next[1])
                r_prev = (0.75 * p[0] + 0.25 * p_prev[0], 0.75 * p[1] + 0.25 * p_prev[1])
                if (self._segment_free(new_pts[-1], r_prev, state)
                        and self._segment_free(r_prev, q, state)
                        and self._segment_free(q, p_next, state)):
                    new_pts.extend([r_prev, q])
                else:
                    new_pts.append(p)
            new_pts.append(pts[-1])
            pts = new_pts
        return pts

    def _resample(self, pts, step):
        out = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            d = math.hypot(b[0] - a[0], b[1] - a[1])
            n = max(int(math.ceil(d / step)), 1)
            for k in range(1, n + 1):
                t = k / n
                out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
        return out

    def smooth_path(self, path):
        """path: list of (x, y, theta, state). Returns list of (x, y, theta)."""
        if len(path) < 3:
            return [(x, y, th) for x, y, th, _ in path]

        # split into runs of constant direction (segment i-1 -> i has direction path[i].state)
        runs = []
        cur = [path[0][:2]]
        cur_state = path[1][3]
        for i in range(1, len(path)):
            s = path[i][3]
            if s != cur_state:
                runs.append((cur_state, cur))
                cur = [path[i - 1][:2]]
                cur_state = s
            cur.append(path[i][:2])
        runs.append((cur_state, cur))

        result = []
        for idx, (state, pts) in enumerate(runs):
            pts = self._shortcut(pts, state)
            pts = self._round_corners(pts, state, self.smoothing_corner_iterations)
            pts = self._resample(pts, self.smoothing_resample_step)

            for k, p in enumerate(pts):
                if idx > 0 and k == 0:
                    continue  # cusp point already added by previous run
                nxt = pts[k + 1] if k + 1 < len(pts) else None
                if nxt is not None:
                    theta = self._segment_heading(p, nxt, state)
                else:
                    theta = result[-1][2] if result else path[0][2]
                result.append((p[0], p[1], theta))

        # keep the true start and goal orientations
        result[0] = (result[0][0], result[0][1], path[0][2])
        result[-1] = (result[-1][0], result[-1][1], path[-1][2])
        return result

    # ------------------------------------------------------------------
    # A*
    # ------------------------------------------------------------------
    def A_star(self, start, goal):
        self.computing = True
        self.open_set_ = _BucketQueue(resolution=self.open_set_bucket_resolution)
        self.closed_set_ = set()

        start_node = node_state(start[0], start[1], start[2], 0, 0, 0, None, 1)
        self.open_set_.push(0.0, start_node)

        best_cost = {}
        best_cost[self.discretize(start_node.x, start_node.y, start_node.theta)] = 0

        expansions = 0
        primitives = self._primitives
        dx_tables = self._dx_table
        dy_tables = self._dy_table

        while self.open_set_:
            current = self.open_set_.pop()
            current_key = self.discretize(current.x, current.y, current.theta)
            if current_key in self.closed_set_:
                continue

            self.closed_set_.add(current_key)
            expansions += 1
            if expansions >= 10000:
                break

            dist_to_goal = math.hypot(current.x - goal[0], current.y - goal[1])
            heading_err = abs(math.atan2(math.sin(current.theta - goal[2]), math.cos(current.theta - goal[2])))

            if dist_to_goal < 0.4 and heading_err < math.radians(25):
                self.get_logger().info("Goal reached!")
                self.get_logger().info(
                    f"Expansions: {expansions}, Open set size: {len(self.open_set_)}, "
                    f"Closed set size: {len(self.closed_set_)}"
                )

                path = []
                node = current
                while node is not None:
                    path.append((node.x, node.y, node.theta, node.state))
                    node = node.parent

                return path[::-1]

            bin_idx = self._theta_to_motion_bin(current.theta)
            dx_row = dx_tables[bin_idx]
            dy_row = dy_tables[bin_idx]

            for p_idx, (v, w, dtheta, move_cost_base, state) in enumerate(primitives):
                x_new = current.x + dx_row[p_idx]
                y_new = current.y + dy_row[p_idx]
                theta_new = self._wrap_angle(current.theta + dtheta)

                collision, c = self._pose_footprint_status(x_new, y_new, theta_new)
                if collision:
                    continue

                if dist_to_goal >= 1.0:
                    h_cost = self.heuristic_cost(x_new, y_new)
                else:
                    h_cost = self.euler_plus_heading(x_new, y_new, theta_new)
                g_cost = current.g + move_cost_base + c

                new_key = self.discretize(x_new, y_new, theta_new)

                if new_key not in best_cost or g_cost < best_cost[new_key]:
                    best_cost[new_key] = g_cost
                    new_node = node_state(x_new, y_new, theta_new, v, w, g_cost, current, state)
                    f_cost = h_cost + g_cost
                    self.open_set_.push(f_cost, new_node)

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
            n_raw = len(path)
            if self.smoothing_enabled:
                t1 = time.perf_counter()
                path = self.smooth_path(path)
                self.get_logger().info(
                    f"Smoothing: {n_raw} raw poses -> {len(path)} poses in {time.perf_counter() - t1:.4f} s"
                )
            else:
                path = [(x, y, th) for x, y, th, _ in path]

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