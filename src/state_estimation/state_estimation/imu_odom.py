#!/usr/bin/env python3
"""IMU-only dead-reckoning odometry (baseline for comparison, NOT fused by the EKF).

Subscribes /imu, publishes /imu/odometry (odom -> base_frame). Gyro-z gives yaw, accelerometer
x/y is double-integrated for position, so position drifts quadratically -- by design.
Keep the robot still for the first `init_duration` seconds (bias calibration).
"""
import math

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu

from .imu_core import ImuDeadReckoning
from .utils import stamp_to_sec


class ImuOdomNode(Node):
    def __init__(self):
        super().__init__('imu_odom')
        self.declare_parameter('imu_topic', '/imu')
        self.declare_parameter('odom_topic', '/imu/odometry')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'zippy')
        self.declare_parameter('init_duration', 1.0)
        self.dr = ImuDeadReckoning(self.get_parameter('init_duration').value)
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.create_subscription(Imu, self.get_parameter('imu_topic').value, self.imu_cb,
                                 qos_profile_sensor_data)
        self.pub = self.create_publisher(Odometry, self.get_parameter('odom_topic').value, 10)

    def imu_cb(self, msg: Imu):
        r = self.dr.add(stamp_to_sec(msg.header.stamp), msg.angular_velocity.z,
                        msg.linear_acceleration.x, msg.linear_acceleration.y)
        if r is None:
            return
        x, y, yaw, vx, wz = r
        o = Odometry()
        o.header.stamp = msg.header.stamp
        o.header.frame_id = self.odom_frame
        o.child_frame_id = self.base_frame
        o.pose.pose.position.x = float(x)
        o.pose.pose.position.y = float(y)
        o.pose.pose.orientation.z = math.sin(0.5 * yaw)
        o.pose.pose.orientation.w = math.cos(0.5 * yaw)
        o.twist.twist.linear.x = float(vx)
        o.twist.twist.angular.z = float(wz)
        pc = [0.0] * 36
        pc[0] = pc[7] = 1.0          # position is poorly known: large variance
        pc[35] = 0.01
        o.pose.covariance = pc
        self.pub.publish(o)


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(ImuOdomNode())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
