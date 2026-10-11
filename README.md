# Zippy

Develop and experimentally evaluate a mapless, GPS-denied point-to-point navigation system in which absolute localization is available only at the start and destination, while the robot autonomously traverses the intervening unknown environment using onboard relative perception, state estimation, planning, and control.

## Architecture (ROS 2)

| Stage | Implementation |
|---|---|
| Relative perception | 2D LiDAR (`/scan`), IMU, wheel odometry |
| State estimation | `robot_localization` EKF fusing encoder velocity + IMU yaw + LIO, `odom -> zippy` (no `map` frame, no SLAM) |
| Planning | None global. The goal is a single point, given relative to the start pose |
| Control | Point-chasing pure pursuit (`local_planners/pure_pursuit.py`) + reactive obstacle avoidance |

The odom origin is the start pose. Convert the destination's absolute position into a start-relative (x, y) offset and publish it on `/goal_pose` (RViz "2D Goal Pose" works with the fixed frame set to `odom`).

## Run

```
colcon build && source install/setup.bash
ros2 launch program_bringup zippy.launch.xml
```

## Odometry stack (state_estimation)

| Source | Topic | What it is |
|---|---|---|
| Encoder | `/odometry` | wheel odometry (`helpers/relay.py`) |
| IMU only | `/imu/odometry` | gyro + double-integrated accelerometer (baseline, drifts fast by design) |
| LIO | `/lio/odometry` | 2D LiDAR-inertial odometry (`lio_node`): IMU prediction + scan-to-local-submap matching, no global map, no loop closure |
| **EKF (fused)** | `/odometry/filtered` | encoder vx/vyaw + IMU yaw + LIO x/y/yaw increments (`config/ekf.yaml`) |

RViz (fixed frame `odom`) shows one path per source on `/viz/path/{encoder,imu,lio,ekf}`:
orange = encoder, magenta = IMU only, cyan = LIO, green (thick) = EKF. `odom_viz` also logs every
source to `~/zippy_odom_logs/odom_<time>.csv`; plot it with `ros2 run state_estimation plot_odometry`.

Keep the robot still for the first ~1 s after launch (IMU bias calibration for LIO and `imu_odom`).

Test without ROS/Gazebo (ray-cast world with ground truth):
`ros2 run state_estimation lio_synthetic_eval --out lio_eval.png` and `pytest src/state_estimation/test`.

### Known limits
- In a long featureless corridor the scan cannot observe along-track motion; LIO falls back to the IMU
  prediction there and drifts along-track (accelerometer bias). Lateral position and yaw stay constrained.
  The encoder velocity in the EKF covers this.
- No motion de-skew of the scan (fine for Gazebo; matters for a real spinning LiDAR at speed).
- The IMU is assumed aligned with the base frame; the robot is assumed to move on a plane.
- EKF noise values (`relay.py` `odom_var_*`, LIO `cov_inflation`, IMU covariance from the sensor) are
  starting points and need tuning on the real data. Check `/imu` covariance is not all zeros: a
  zero covariance makes robot_localization treat the IMU as perfect.
