#!/usr/bin/env python3
"""Odometry visualization: one nav_msgs/Path per odometry source, plus an optional CSV log.

Default sources (name:topic), each shown as its own coloured path in RViz:
  encoder  /odometry            wheel encoders (helpers/relay.py)
  imu      /imu/odometry        IMU-only dead reckoning (imu_odom)
  lio      /lio/odometry        LiDAR-inertial odometry (lio_node)
  ekf      /odometry/filtered   robot_localization EKF fusing encoder + IMU + LIO
Paths are published on <path_topic_prefix>/<name> (default /viz/path/<name>), frame `odom`.
Every source starts at the origin (odom = start pose), so the curves are directly comparable.

CSV: <log_dir>/odom_<timestamp>.csv with columns t,source,x,y,yaw -> plot with plot_odometry.
"""
import csv
import datetime
import functools
import math
import os

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node

from .utils import stamp_to_sec, wrap, yaw_from_quat

DEFAULT_SOURCES = ['encoder:/odometry', 'imu:/imu/odometry', 'lio:/lio/odometry',
                   'ekf:/odometry/filtered']


class OdomVizNode(Node):
    def __init__(self):
        super().__init__('odom_viz')
        self.declare_parameter('sources', DEFAULT_SOURCES)
        self.declare_parameter('path_topic_prefix', '/viz/path')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('min_spacing', 0.03)       # m between stored path poses
        self.declare_parameter('min_yaw_step', 0.05)      # rad
        self.declare_parameter('max_poses', 4000)         # per path (oldest dropped)
        self.declare_parameter('publish_rate', 2.0)       # Hz
        self.declare_parameter('log_dir', '~/zippy_odom_logs')   # '' disables the CSV log
        self.declare_parameter('log_rate', 20.0)          # max CSV rows per second per source

        g = lambda n: self.get_parameter(n).value
        self.frame = g('odom_frame')
        self.min_spacing, self.min_yaw, self.max_poses = g('min_spacing'), g('min_yaw_step'), int(g('max_poses'))
        self.log_dt = 1.0 / max(g('log_rate'), 1e-3)

        self.poses, self.stamps, self.last, self.last_log, self.pubs = {}, {}, {}, {}, {}
        for spec in g('sources'):
            name, topic = spec.split(':', 1)
            self.poses[name] = []
            self.pubs[name] = self.create_publisher(Path, f"{g('path_topic_prefix')}/{name}", 10)
            self.create_subscription(Odometry, topic, functools.partial(self.odom_cb, name), 50)

        self.csv_file = self.csv = None
        if g('log_dir'):
            d = os.path.expanduser(g('log_dir'))
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, 'odom_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.csv')
            self.csv_file = open(path, 'w', newline='')
            self.csv = csv.writer(self.csv_file)
            self.csv.writerow(['t', 'source', 'x', 'y', 'yaw'])
            self.get_logger().info(f'logging odometry to {path}')
        self.create_timer(1.0 / max(g('publish_rate'), 1e-3), self.publish_paths)
        self.get_logger().info('odom_viz paths: ' + ', '.join(f"{g('path_topic_prefix')}/{n}" for n in self.poses))

    def odom_cb(self, name, msg: Odometry):
        p = msg.pose.pose
        x, y, yaw = p.position.x, p.position.y, yaw_from_quat(p.orientation)
        t = stamp_to_sec(msg.header.stamp)
        self.stamps[name] = msg.header.stamp
        last = self.last.get(name)
        if last is None or math.hypot(x - last[0], y - last[1]) >= self.min_spacing \
                or abs(wrap(yaw - last[2])) >= self.min_yaw:
            ps = PoseStamped()
            ps.header.stamp = msg.header.stamp
            ps.header.frame_id = self.frame
            ps.pose.position.x, ps.pose.position.y = x, y
            ps.pose.orientation.z, ps.pose.orientation.w = p.orientation.z, p.orientation.w
            lst = self.poses[name]
            lst.append(ps)
            if len(lst) > self.max_poses:
                del lst[0]
            self.last[name] = (x, y, yaw)
        if self.csv is not None and t - self.last_log.get(name, -1e9) >= self.log_dt:
            self.last_log[name] = t
            self.csv.writerow([f'{t:.4f}', name, f'{x:.5f}', f'{y:.5f}', f'{yaw:.5f}'])

    def publish_paths(self):
        for name, lst in self.poses.items():
            if not lst:
                continue
            path = Path()
            path.header.frame_id = self.frame
            path.header.stamp = self.stamps[name]
            path.poses = list(lst)
            self.pubs[name].publish(path)
        if self.csv_file is not None:
            self.csv_file.flush()

    def destroy_node(self):
        if self.csv_file is not None:
            self.csv_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = OdomVizNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
