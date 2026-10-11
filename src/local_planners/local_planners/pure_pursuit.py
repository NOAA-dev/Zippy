#!/usr/bin/env python3
import math

import numpy as np
import rclpy
from geometry_msgs.msg import Twist, PoseStamped
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener, TransformException
from tf_transformations import euler_from_quaternion


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class PurePursuitNode(Node):
    def __init__(self):
        super().__init__("pure_pursuit_node")

        # ---- parameters ----
        p = self.declare_parameter
        p("world_frame", "odom")
        p("robot_frame", "zippy")
        p("base_velocity", 0.8)
        p("min_velocity", 0.05)
        p("max_angular_vel", 1.0)
        p("min_angular_vel", 0.08)          # rad/s, overcomes motor deadband while aligning
        p("tf_timeout_sec", 0.5)

        # goal convergence
        p("position_tolerance", 0.23)       # m   (final stop radius)
        p("yaw_tolerance", 0.1)            
        p("goal_slowdown_distance", 1.5)
        p("final_approach_distance", 0.30)  # m, switch from arc-following to proportional approach
        p("final_max_velocity", 0.20)
        p("final_linear_gain", 1.0)
        p("final_realign_angle", 0.5)       # rad, in FINAL rotate in place if heading error exceeds this
        p("rotate_in_place_angle", 1.0)     # rad
        p("heading_gain", 1.5)

        # footprint (robot_frame origin assumed at footprint centre unless offset given)
        p("robot_length", 0.60)             # along x
        p("robot_width", 0.50)              # along y
        p("footprint_center_x", 0.0)        # footprint centre relative to robot_frame origin (+x forward)
        p("safety_margin", 0.05)            # extra lateral margin for the swept corridor
        p("side_lookahead", 0.60)           # how far ahead of the front edge side obstacles still count
        p("laser_x", 0.0)                   # laser pose in robot_frame
        p("laser_y", 0.0)
        p("laser_yaw", 0.0)

        # avoidance
        p("avoid_start_distance", 1.20)     # clearance (from footprint edge) where avoidance starts
        p("avoid_stop_distance", 0.15)      # clearance where the robot stops
        p("clear_default", 10.0)
        p("check_rotation_clearance", True)
        p("rotation_margin", 0.03)

        g = lambda n: self.get_parameter(n).value
        self.world_frame = g("world_frame")
        self.robot_frame = g("robot_frame")
        self.base_velocity = g("base_velocity")
        self.min_velocity = g("min_velocity")
        self.max_angular_vel = g("max_angular_vel")
        self.min_angular_vel = g("min_angular_vel")
        self.tf_timeout_sec = g("tf_timeout_sec")
        self.pos_tol = g("position_tolerance")
        self.yaw_tol = g("yaw_tolerance")
        self.goal_slowdown_distance = g("goal_slowdown_distance")
        self.final_dist = g("final_approach_distance")
        self.final_max_v = g("final_max_velocity")
        self.final_kp = g("final_linear_gain")
        self.final_realign_angle = g("final_realign_angle")
        self.rotate_in_place_angle = g("rotate_in_place_angle")
        self.heading_gain = g("heading_gain")

        length, width = g("robot_length"), g("robot_width")
        cx = g("footprint_center_x")
        self.front_extent = length / 2.0 + cx      # origin -> front edge
        self.rear_extent = length / 2.0 - cx       # origin -> rear edge
        self.half_width = width / 2.0
        self.safety_margin = g("safety_margin")
        self.side_lookahead = g("side_lookahead")
        self.laser_x, self.laser_y, self.laser_yaw = g("laser_x"), g("laser_y"), g("laser_yaw")
        # radius of the circle swept when rotating in place about the frame origin
        self.circum_radius = math.hypot(max(self.front_extent, self.rear_extent), self.half_width)

        self.avoid_start_distance = g("avoid_start_distance")
        self.avoid_stop_distance = g("avoid_stop_distance")
        self.clear_default = g("clear_default")
        self.check_rotation = g("check_rotation_clearance")
        self.rotation_margin = g("rotation_margin")

        # ---- robot pose (from TF) ----
        self.x_pos = 0.0
        self.y_pos = 0.0
        self.yaw = 0.0
        self.v = 0.0
        self.last_tf_time = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_timer = self.create_timer(0.02, self.check_tf)

        # ---- goal pose (world_frame) ----
        self.goal_x = None
        self.goal_y = None
        self.goal_yaw = None
        self.state = "DONE"
        self.goal_sub = self.create_subscription(PoseStamped, "/goal_pose", self.goal_callback, 10)

        # ---- obstacle data ----
        self.front_clearance = self.clear_default
        self.right_clearance = self.clear_default
        self.left_clearance = self.clear_default
        self.scan_xy = np.empty((0, 2))
        self.subscriber_ = self.create_subscription(LaserScan, "/scan", self.laser_callback, 10)

        self.publisher_ = self.create_publisher(Twist, "/autonomous_cmd_vel", 10)
        self.timer_ = self.create_timer(0.02, self.pure_pursuit)  # 50 Hz

    # ------------------------------------------------------------------
    def check_tf(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.world_frame, self.robot_frame, rclpy.time.Time(), timeout=Duration(seconds=0.2))
            self.x_pos = transform.transform.translation.x
            self.y_pos = transform.transform.translation.y
            q = transform.transform.rotation
            _, _, self.yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
            self.last_tf_time = self.get_clock().now()
        except TransformException as ex:
            self.get_logger().warn(f"TF unavailable: {ex}", throttle_duration_sec=2.0)

    def tf_is_stale(self):
        if self.last_tf_time is None:
            return True
        age = (self.get_clock().now() - self.last_tf_time).nanoseconds / 1e9
        return age > self.tf_timeout_sec

    # ------------------------------------------------------------------
    def laser_callback(self, msg: LaserScan):
        r = np.asarray(msg.ranges, dtype=float)
        a = msg.angle_min + np.arange(r.size) * msg.angle_increment
        ok = np.isfinite(r) & (r >= max(msg.range_min, 0.01)) & (r <= msg.range_max)
        r, a = r[ok], a[ok]

        # scan points in the robot frame
        ang = a + self.laser_yaw
        x = self.laser_x + r * np.cos(ang)
        y = self.laser_y + r * np.sin(ang)
        self.scan_xy = np.column_stack((x, y))

        # Front: anything inside the swept corridor (footprint width + margin) ahead of the front edge.
        corridor = self.half_width + self.safety_margin
        m = (x > 0.0) & (np.abs(y) <= corridor)
        self.front_clearance = (max(0.0, float(np.min(x[m])) - self.front_extent)
                                if m.any() else self.clear_default)

        # Sides: lateral gap to the footprint edge for points alongside the body (or just ahead of it).
        along = (x >= -self.rear_extent) & (x <= self.front_extent + self.side_lookahead)
        ml = along & (y > 0.0)
        mr = along & (y < 0.0)
        self.left_clearance = (max(0.0, float(np.min(y[ml])) - self.half_width)
                               if ml.any() else self.clear_default)
        self.right_clearance = (max(0.0, float(np.min(-y[mr])) - self.half_width)
                                if mr.any() else self.clear_default)

    def rotation_blocked(self):
        """True if something lies inside the circle swept by the footprint when spinning in place."""
        if not self.check_rotation or self.scan_xy.size == 0:
            return False
        d = np.hypot(self.scan_xy[:, 0], self.scan_xy[:, 1])
        return bool(np.any(d < self.circum_radius + self.rotation_margin))

    # ------------------------------------------------------------------
    def goal_callback(self, msg: PoseStamped):
        """Accept a goal pose; convert into world_frame if it arrives in another frame."""
        x = msg.pose.position.x
        y = msg.pose.position.y
        q = msg.pose.orientation
        has_yaw = (q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w) > 1e-6
        gyaw = euler_from_quaternion([q.x, q.y, q.z, q.w])[2] if has_yaw else None

        frame = msg.header.frame_id
        if frame and frame != self.world_frame:
            try:
                t = self.tf_buffer.lookup_transform(
                    self.world_frame, frame, rclpy.time.Time(), timeout=Duration(seconds=0.5))
            except TransformException as ex:
                self.get_logger().error(f"Goal rejected, cannot transform '{frame}' -> '{self.world_frame}': {ex}")
                return
            tq = t.transform.rotation
            _, _, tyaw = euler_from_quaternion([tq.x, tq.y, tq.z, tq.w])
            x, y = (t.transform.translation.x + math.cos(tyaw) * x - math.sin(tyaw) * y,
                    t.transform.translation.y + math.sin(tyaw) * x + math.cos(tyaw) * y)
            if gyaw is not None:
                gyaw = wrap(gyaw + tyaw)

        self.goal_x, self.goal_y, self.goal_yaw = x, y, gyaw
        self.state = "DRIVE"
        self.v = 0.0
        yaw_txt = f"{math.degrees(gyaw):.1f} deg" if gyaw is not None else "none (position only)"
        self.get_logger().info(f"New goal ({x:.2f}, {y:.2f}), yaw {yaw_txt} in '{self.world_frame}'")

    # ------------------------------------------------------------------
    def publish_cmd(self, v=0.0, w=0.0):
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.publisher_.publish(msg)

    def rotate_in_place(self, error, gain):
        """Rotate toward `error` (rad). Returns without moving if the spin sweep is blocked."""
        self.v = 0.0
        if self.rotation_blocked():
            self.get_logger().warn("Rotation blocked by obstacle inside footprint sweep - holding",
                                   throttle_duration_sec=1.0)
            self.publish_cmd()
            return
        omega = float(np.clip(gain * error, -self.max_angular_vel, self.max_angular_vel))
        if abs(omega) < self.min_angular_vel:
            omega = math.copysign(self.min_angular_vel, error)
        self.publish_cmd(0.0, omega)

    # ------------------------------------------------------------------
    def pure_pursuit(self):
        if self.goal_x is None or self.state == "DONE":
            return

        if self.tf_is_stale():
            self.get_logger().warn("TF stale - holding robot", throttle_duration_sec=1.0)
            self.publish_cmd()
            return

        dx = self.goal_x - self.x_pos
        dy = self.goal_y - self.y_pos
        dist = math.hypot(dx, dy)

        # ---------------- ALIGN: position reached, now match the goal orientation ----------------
        if self.state == "ALIGN":
            if dist > 2.0 * self.pos_tol:          # drifted away -> go back and fix position
                self.state = "DRIVE"
            else:
                if self.goal_yaw is None:
                    self.finish()
                    return
                yaw_err = wrap(self.goal_yaw - self.yaw)
                if abs(yaw_err) <= self.yaw_tol:
                    self.finish()
                    return
                self.rotate_in_place(yaw_err, self.heading_gain)
                return

        # position reached -> stop translating and start aligning
        if dist <= self.pos_tol:
            self.v = 0.0
            self.state = "ALIGN"
            self.publish_cmd()
            self.get_logger().info("Position reached - aligning to goal yaw")
            return

        # goal point in the robot frame
        c, s = math.cos(-self.yaw), math.sin(-self.yaw)
        x_r = c * dx - s * dy
        y_r = s * dx + c * dy
        alpha = math.atan2(y_r, x_r)

        # ---------------- FINAL: slow proportional approach, no arc tracking ----------------
        if dist <= self.final_dist:
            self.state = "FINAL"
            if abs(alpha) > self.final_realign_angle:   # overshot / badly aimed -> face the point first
                self.rotate_in_place(alpha, self.heading_gain)
                return
            target_v = float(np.clip(self.final_kp * dist, self.min_velocity, self.final_max_v))
            self.v += (target_v - self.v) * 0.7
            omega = float(np.clip(self.heading_gain * alpha, -self.max_angular_vel, self.max_angular_vel))
            # Obstacle avoidance is not applied here: the goal may legitimately be close to an obstacle.
            self.publish_cmd(self.v, omega)
            return
        self.state = "DRIVE"

        # ---------------- DRIVE: pure pursuit arc ----------------
        if abs(alpha) > self.rotate_in_place_angle:
            self.rotate_in_place(alpha, self.heading_gain)
            return

        target_v = self.base_velocity * min(dist / self.goal_slowdown_distance, 1.0)
        target_v = max(target_v, self.min_velocity)
        self.v += (target_v - self.v) * 0.7

        curvature = 2.0 * y_r / (dist ** 2)
        omega = float(np.clip(self.v * curvature, -self.max_angular_vel, self.max_angular_vel))

        # reactive avoidance using footprint-based clearances
        span = max(self.avoid_start_distance - self.avoid_stop_distance, 1e-6)
        v_scale = np.clip((self.front_clearance - self.avoid_stop_distance) / span, 0.0, 1.0)
        self.v *= v_scale

        nearest = min(self.front_clearance, self.left_clearance, self.right_clearance)
        if nearest < self.avoid_start_distance:
            urgency = np.clip((self.avoid_start_distance - nearest) / span, 0.0, 1.0)
            side_diff = np.clip(self.left_clearance - self.right_clearance, -1.0, 1.0)
            if abs(side_diff) < 0.1:               # symmetric: break the tie toward the goal side
                side_diff = 1.0 if y_r >= 0.0 else -1.0
            avoid_omega = urgency * side_diff * self.max_angular_vel
            omega = (1 - urgency) * omega + urgency * avoid_omega

        omega = float(np.clip(omega, -self.max_angular_vel, self.max_angular_vel))
        self.publish_cmd(self.v, omega)

    def finish(self):
        self.state = "DONE"
        self.v = 0.0
        self.publish_cmd()
        yaw_txt = ""
        if self.goal_yaw is not None:
            yaw_txt = f", yaw error {math.degrees(wrap(self.goal_yaw - self.yaw)):.2f} deg"
        self.get_logger().info(
            f"Goal reached: position error {math.hypot(self.goal_x - self.x_pos, self.goal_y - self.y_pos):.3f} m{yaw_txt}")


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()