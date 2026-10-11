#!/usr/bin/env python3
"""ROS 2 node for 2D LiDAR-inertial odometry. Algorithm lives in lio_core.py.

Subscribes : /scan (LaserScan), /imu (Imu)
Publishes  : /lio/odometry (nav_msgs/Odometry, odom -> base_frame). No TF is broadcast: the EKF
             owns odom -> base_frame.

The pose covariance is the PER-STEP uncertainty of (x, y, yaw), not an accumulated one. That is
what robot_localization expects for a measurement fused with `odom1_differential: true`.
The robot must be stationary for the first `init_duration` seconds (IMU bias calibration).
"""
import dataclasses
import math

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, LaserScan
from tf2_ros import Buffer, TransformException, TransformListener

from .lio_core import Lio2D, LioParams, scan_to_points
from .utils import stamp_to_sec, yaw_from_quat


class LioNode(Node):
    def __init__(self):
        super().__init__('lio_node')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('imu_topic', '/imu')
        self.declare_parameter('odom_topic', '/lio/odometry')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'zippy')
        self.declare_parameter('extrinsic_from_tf', True)
        self.declare_parameter('lidar_x', 0.0)          # used only if extrinsic_from_tf is False
        self.declare_parameter('lidar_y', 0.0)
        self.declare_parameter('lidar_yaw', 0.0)

        kw = {}
        for f in dataclasses.fields(LioParams):
            if f.name == 'lidar_to_base':
                continue
            self.declare_parameter(f.name, f.default)
            kw[f.name] = self.get_parameter(f.name).value
        self.lio = Lio2D(LioParams(**kw))

        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.extrinsic_ready = False
        if not self.get_parameter('extrinsic_from_tf').value:
            self.lio.set_lidar_extrinsic(self.get_parameter('lidar_x').value,
                                         self.get_parameter('lidar_y').value,
                                         self.get_parameter('lidar_yaw').value)
            self.extrinsic_ready = True

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._last_pub_t = None
        self._calib_logged = False

        # sensor_data QoS (best effort) connects to both the Gazebo bridge and real drivers
        self.create_subscription(Imu, self.get_parameter('imu_topic').value, self.imu_cb,
                                 qos_profile_sensor_data)
        self.create_subscription(LaserScan, self.get_parameter('scan_topic').value, self.scan_cb,
                                 qos_profile_sensor_data)
        self.pub = self.create_publisher(Odometry, self.get_parameter('odom_topic').value, 10)
        self.get_logger().info('LIO started: keep the robot still for the first '
                               f'{self.lio.p.init_duration:.1f} s of IMU data')

    # ------------------------------------------------------------------
    def imu_cb(self, msg: Imu):
        self.lio.add_imu(stamp_to_sec(msg.header.stamp), msg.angular_velocity.z,
                         msg.linear_acceleration.x, msg.linear_acceleration.y)
        if not self._calib_logged and self.lio.calib_ok is not None:
            self._calib_logged = True
            if self.lio.calib_ok:
                self.get_logger().info(f'IMU calibrated: gyro bias {self.lio.bg:.4f} rad/s, '
                                       f'accel bias ({self.lio.ba[0]:.3f}, {self.lio.ba[1]:.3f}) m/s^2')
            else:
                self.get_logger().warn('Robot was moving during IMU calibration; biases left at zero')

    def _load_extrinsic(self, scan_frame):
        try:
            t = self.tf_buffer.lookup_transform(self.base_frame, scan_frame, rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().warn(f'Waiting for TF {self.base_frame} -> {scan_frame}: {ex}',
                                   throttle_duration_sec=2.0)
            return False
        self.lio.set_lidar_extrinsic(t.transform.translation.x, t.transform.translation.y,
                                     yaw_from_quat(t.transform.rotation))
        self.extrinsic_ready = True
        self.get_logger().info(f'Lidar extrinsic from TF: ({t.transform.translation.x:.3f}, '
                               f'{t.transform.translation.y:.3f}, yaw {yaw_from_quat(t.transform.rotation):.3f})')
        return True

    def scan_cb(self, msg: LaserScan):
        if not self.extrinsic_ready and not self._load_extrinsic(msg.header.frame_id):
            return
        p = self.lio.p
        pts = scan_to_points(msg.ranges, msg.angle_min, msg.angle_increment,
                             max(p.range_min, msg.range_min), min(p.range_max, msg.range_max))
        res = self.lio.process_scan(stamp_to_sec(msg.header.stamp), pts)
        if res is None:
            return
        self._publish(msg.header.stamp, res)
        if not res.ok:
            self.get_logger().warn('scan match rejected - using IMU prediction only',
                                   throttle_duration_sec=2.0)
        self.get_logger().info(f'pose ({res.pose[0]:.2f}, {res.pose[1]:.2f}, {res.pose[2]:.2f}) '
                               f'inliers {res.inlier_ratio:.2f} weakest-constraint '
                               f"{res.info.get('lam_min', 0.0):.0f} gyro-bias {res.gyro_bias:.4f}",
                               throttle_duration_sec=5.0)

    def _publish(self, stamp, res):
        o = Odometry()
        o.header.stamp = stamp
        o.header.frame_id = self.odom_frame
        o.child_frame_id = self.base_frame
        o.pose.pose.position.x = float(res.pose[0])
        o.pose.pose.position.y = float(res.pose[1])
        o.pose.pose.orientation.z = math.sin(0.5 * res.pose[2])
        o.pose.pose.orientation.w = math.cos(0.5 * res.pose[2])

        C = res.cov
        pc = [0.0] * 36
        for i, ii in enumerate((0, 1, 5)):               # x, y, yaw rows/cols of the 6x6
            for j, jj in enumerate((0, 1, 5)):
                pc[6 * ii + jj] = float(C[i, j])
        for k in (14, 21, 28):                           # z, roll, pitch: not observed
            pc[k] = 1e6
        o.pose.covariance = pc

        t = stamp_to_sec(stamp)
        dt = (t - self._last_pub_t) if self._last_pub_t is not None else 0.1
        self._last_pub_t = t
        dt = max(dt, 1e-3)
        o.twist.twist.linear.x = float(res.vel[0])
        o.twist.twist.linear.y = float(res.vel[1])
        o.twist.twist.angular.z = float(res.vel[2])
        tc = [0.0] * 36
        tc[0] = float(C[0, 0]) / dt ** 2
        tc[7] = float(C[1, 1]) / dt ** 2
        tc[35] = float(C[2, 2]) / dt ** 2
        for k in (14, 21, 28):
            tc[k] = 1e6
        o.twist.covariance = tc
        self.pub.publish(o)


def main(args=None):
    rclpy.init(args=args)
    node = LioNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
