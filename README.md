# Zippy

Develop and experimentally evaluate a mapless, GPS-denied point-to-point navigation system in which absolute localization is available only at the start and destination, while the robot autonomously traverses the intervening unknown environment using onboard relative perception, state estimation, planning, and control.

## Architecture (ROS 2)

| Stage | Implementation |
|---|---|
| Relative perception | 2D LiDAR (`/scan`), IMU, wheel odometry |
| State estimation | `robot_localization` EKF, `odom -> zippy` (no `map` frame, no SLAM) |
| Planning | None global. The goal is a single point, given relative to the start pose |
| Control | Point-chasing pure pursuit (`local_planners/pure_pursuit.py`) + reactive obstacle avoidance |

The odom origin is the start pose. Convert the destination's absolute position into a start-relative (x, y) offset and publish it on `/goal_pose` (RViz "2D Goal Pose" works with the fixed frame set to `odom`).

## Run

```
colcon build && source install/setup.bash
ros2 launch program_bringup zippy.launch.xml
```
