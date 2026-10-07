#!/usr/bin/env python3
"""Point-chasing pure pursuit.

The robot steers along the circular arc that joins its current pose to a single
goal point (the "lookahead point" IS the goal). There is no path, no global
planner and no map: the pose comes from odometry/EKF (odom -> zippy) and the
goal is a point expressed relative to the start pose (odom origin = start).
"""
import rclpy
import numpy as np
import math
from geometry_msgs.msg import Twist, PoseStamped
from tf2_ros import Buffer, TransformListener, TransformException
from rclpy.duration import Duration
from rclpy.node import Node
from tf_transformations import euler_from_quaternion
from sensor_msgs.msg import LaserScan


class PurePursuitNode(Node):
    def __init__(self):
        super().__init__("pure_pursuit_node")

        # ---- parameters ----
        self.declare_parameter("world_frame", "odom")
        self.declare_parameter("robot_frame", "zippy")
        self.declare_parameter("base_velocity", 0.8)
        self.declare_parameter("min_velocity", 0.05)
        self.declare_parameter("max_angular_vel", 1.0)
        self.declare_parameter("tf_timeout_sec", 0.5)

        self.declare_parameter("goal_tolerance", 0.2)
        self.declare_parameter("goal_slowdown_distance", 1.5)
        self.declare_parameter("rotate_in_place_angle", 1.0)   # rad (~57 deg)
        self.declare_parameter("heading_gain", 1.5)

        self.declare_parameter("avoid_start_distance", 0.6)
        self.declare_parameter("avoid_stop_distance", 0.25)
        self.declare_parameter("clear_default", 10.0)
        self.declare_parameter("front_clearance_offset", 0.3)
        self.declare_parameter("side_clearance_offset", 0.225)

        self.world_frame = self.get_parameter("world_frame").value
        self.robot_frame = self.get_parameter("robot_frame").value
        self.base_velocity = self.get_parameter("base_velocity").value
        self.min_velocity = self.get_parameter("min_velocity").value
        self.max_angular_vel = self.get_parameter("max_angular_vel").value
        self.tf_timeout_sec = self.get_parameter("tf_timeout_sec").value
        self.goal_tolerance = self.get_parameter("goal_tolerance").value
        self.goal_slowdown_distance = self.get_parameter("goal_slowdown_distance").value
        self.rotate_in_place_angle = self.get_parameter("rotate_in_place_angle").value
        self.heading_gain = self.get_parameter("heading_gain").value
        self.avoid_start_distance = self.get_parameter("avoid_start_distance").value
        self.avoid_stop_distance = self.get_parameter("avoid_stop_distance").value
        self.clear_default = self.get_parameter("clear_default").value
        self.front_clearance_offset = self.get_parameter("front_clearance_offset").value
        self.side_clearance_offset = self.get_parameter("side_clearance_offset").value

        # ---- robot pose (from TF) ----
        self.x_pos = 0.0
        self.y_pos = 0.0
        self.yaw = 0.0
        self.v = 0.0
        self.last_tf_time = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_timer = self.create_timer(0.02, self.check_tf)

        # ---- goal point (world_frame) ----
        self.goal_x = None
        self.goal_y = None
        self.goal_reached = False
        self.goal_sub = self.create_subscription(PoseStamped, "/goal_pose", self.goal_callback, 10)

        # ---- obstacle clearance ----
        self.front_clearance = self.clear_default
        self.right_clearance = self.clear_default
        self.left_clearance = self.clear_default
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
        angle_min = msg.angle_min
        angle_inc = msg.angle_increment
        ranges = msg.ranges

        front_range = []
        right_range = []
        left_range = []

        angle = angle_min
        front_angle = np.deg2rad(35)
        left_angle = np.deg2rad(120)
        right_angle = np.deg2rad(-120)
        for r in ranges:
            if math.isfinite(r):
                if right_angle <= angle <= -front_angle:
                    right_range.append(r)
                if -front_angle <= angle <= front_angle:
                    front_range.append(r)
                if front_angle <= angle <= left_angle:
                    left_range.append(r)
            angle += angle_inc

        front_min = sum(sorted(front_range)[:10]) / min(len(front_range), 10) if front_range else None
        right_min = sum(sorted(right_range)[:10]) / min(len(right_range), 10) if right_range else None
        left_min = sum(sorted(left_range)[:10]) / min(len(left_range), 10) if left_range else None

        self.front_clearance = max(0.0, front_min - self.front_clearance_offset) if front_min is not None else self.clear_default
        self.right_clearance = max(0.0, right_min - self.side_clearance_offset) if right_min is not None else self.clear_default
        self.left_clearance = max(0.0, left_min - self.side_clearance_offset) if left_min is not None else self.clear_default

    # ------------------------------------------------------------------
    def goal_callback(self, msg: PoseStamped):
        """Accept a goal point; convert it into world_frame if it arrives in another frame."""
        x = msg.pose.position.x
        y = msg.pose.position.y

        frame = msg.header.frame_id
        if frame and frame != self.world_frame:
            try:
                t = self.tf_buffer.lookup_transform(
                    self.world_frame, frame, rclpy.time.Time(), timeout=Duration(seconds=0.5))
            except TransformException as ex:
                self.get_logger().error(f"Goal rejected, cannot transform '{frame}' -> '{self.world_frame}': {ex}")
                return
            q = t.transform.rotation
            _, _, tyaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
            x, y = (t.transform.translation.x + math.cos(tyaw) * x - math.sin(tyaw) * y,
                    t.transform.translation.y + math.sin(tyaw) * x + math.cos(tyaw) * y)

        self.goal_x, self.goal_y = x, y
        self.goal_reached = False
        self.get_logger().info(f"New goal ({x:.2f}, {y:.2f}) in '{self.world_frame}'")

    # ------------------------------------------------------------------
    def pure_pursuit(self):
        if self.goal_x is None or self.goal_reached:
            return

        if self.tf_is_stale():
            self.get_logger().warn("TF stale - holding robot", throttle_duration_sec=1.0)
            self.publisher_.publish(Twist())
            return

        dx = self.goal_x - self.x_pos
        dy = self.goal_y - self.y_pos
        dist_to_goal = math.hypot(dx, dy)

        if dist_to_goal < self.goal_tolerance:
            self.goal_reached = True
            self.v = 0.0
            self.publisher_.publish(Twist())
            self.get_logger().info("Goal reached")
            return

        # goal point in the robot frame
        x_r = math.cos(-self.yaw) * dx - math.sin(-self.yaw) * dy
        y_r = math.sin(-self.yaw) * dx + math.cos(-self.yaw) * dy
        alpha = math.atan2(y_r, x_r)  # heading error to the goal point

        # Goal far off to the side/behind: an arc to it is not drivable, so turn in place first.
        if abs(alpha) > self.rotate_in_place_angle:
            self.v = 0.0
            omega = float(np.clip(self.heading_gain * alpha, -self.max_angular_vel, self.max_angular_vel))
            msg = Twist()
            msg.angular.z = omega
            self.publisher_.publish(msg)
            return

        # speed: slow down on approach to the goal point
        target_v = self.base_velocity * min(dist_to_goal / self.goal_slowdown_distance, 1.0)
        target_v = max(target_v, self.min_velocity)
        self.v += (target_v - self.v) * 0.7

        # pure pursuit toward the goal point: arc curvature = 2*y / d^2
        curvature = 2.0 * y_r / (dist_to_goal ** 2)
        omega = float(np.clip(self.v * curvature, -self.max_angular_vel, self.max_angular_vel))

        # reactive obstacle avoidance (unchanged): slow down and steer toward the clearer side
        v_scale = np.clip((self.front_clearance - self.avoid_stop_distance)/max(self.avoid_start_distance - self.avoid_stop_distance, 1e-6), 0.0, 1.0)
        self.v *= v_scale

        nearest_clearance = min(self.front_clearance, self.left_clearance, self.right_clearance)
        if nearest_clearance < self.avoid_start_distance:
            urgency = np.clip((self.avoid_start_distance - nearest_clearance)/max(self.avoid_start_distance - self.avoid_stop_distance, 1e-6), 0.0, 1.0)
            clearance_error = self.left_clearance - self.right_clearance
            avoid_omega = urgency * np.clip(clearance_error, -1.0, 1.0) * self.max_angular_vel
            omega = (1 - urgency) * omega + urgency * avoid_omega

        omega = float(np.clip(omega, -self.max_angular_vel, self.max_angular_vel))

        msg = Twist()
        msg.linear.x = self.v
        msg.angular.z = omega
        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
