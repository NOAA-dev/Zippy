"""Small ROS-free helpers shared by the nodes."""
import math


def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi
