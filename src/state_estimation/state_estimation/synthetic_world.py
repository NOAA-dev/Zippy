"""Synthetic 2D world for testing odometry without ROS/Gazebo.

A ray-cast arena with box/cylinder obstacles, a unicycle ground-truth trajectory, and
simulated IMU (gyro/accel with bias + noise), 2D LiDAR (Gaussian range noise) and wheel
encoders (scale error, yaw bias, noise) so LIO / IMU / encoder odometry can be compared
against known ground truth.
"""
import math

import numpy as np

from .lio_core import compose, inverse, wrap, scan_to_points


# ----------------------------------------------------------------------------- trajectory
def make_trajectory(duration=150.0, rate=100.0, start=(-6.0, -6.0, 0.0)):
    """Unicycle ground truth sampled at `rate` Hz: waypoint-following loop inside the arena.

    Stationary for the first 2 s, then ramps up. Speed/turn-rate are low-pass filtered so the
    accelerations are smooth like a real robot's.
    """
    wps = [(5, -6), (6.5, 0), (5, 6), (0, 3), (-5, 6), (-6.5, 0), (-5, -6), (0, -3)]
    n = int(duration * rate)
    dt = 1.0 / rate
    t = np.arange(n) * dt
    x, y, yaw = np.empty(n), np.empty(n), np.empty(n)
    v, w = np.zeros(n), np.zeros(n)
    x[0], y[0], yaw[0] = start
    k = 0
    for i in range(1, n):
        ti = t[i]
        if math.hypot(wps[k][0] - x[i - 1], wps[k][1] - y[i - 1]) < 1.0:
            k = (k + 1) % len(wps)
        err = wrap(math.atan2(wps[k][1] - y[i - 1], wps[k][0] - x[i - 1]) - yaw[i - 1])
        ramp = min(max((ti - 2.0) / 3.0, 0.0), 1.0)
        v_cmd = ramp * (0.55 + 0.15 * math.sin(2 * math.pi * ti / 23.0)) * max(0.35, math.cos(err))
        w_cmd = ramp * float(np.clip(1.5 * err, -0.8, 0.8))
        v[i] = v[i - 1] + (v_cmd - v[i - 1]) * dt / 0.5
        w[i] = w[i - 1] + (w_cmd - w[i - 1]) * dt / 0.4
        ym = yaw[i - 1] + 0.5 * w[i - 1] * dt
        x[i] = x[i - 1] + v[i - 1] * math.cos(ym) * dt
        y[i] = y[i - 1] + v[i - 1] * math.sin(ym) * dt
        yaw[i] = wrap(yaw[i - 1] + w[i - 1] * dt)
    a_long = np.gradient(v, dt)
    return dict(t=t, x=x, y=y, yaw=yaw, v=v, w=w, a_long=a_long, rate=rate)


# ----------------------------------------------------------------------------- world
def _box(cx, cy, hx, hy):
    c = [(cx - hx, cy - hy), (cx + hx, cy - hy), (cx + hx, cy + hy), (cx - hx, cy + hy)]
    return [(*c[i], *c[(i + 1) % 4]) for i in range(4)]


def _cyl(cx, cy, r, n=14):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    p = [(cx + r * math.cos(k), cy + r * math.sin(k)) for k in a]
    return [(*p[i], *p[(i + 1) % n]) for i in range(n)]


def make_world(traj, half=10.0, n_obstacles=16, clearance=1.1, seed=0):
    """Square arena + random obstacles kept at least `clearance` m away from the path."""
    rng = np.random.default_rng(seed)
    segs = _box(0.0, 0.0, half, half)
    path = np.column_stack([traj['x'], traj['y']])[::20]
    centers = []
    tries = 0
    while len(centers) < n_obstacles and tries < 5000:
        tries += 1
        c = rng.uniform(-half + 1.5, half - 1.5, 2)
        size = rng.uniform(0.3, 0.8)
        if np.min(np.hypot(*(path - c).T)) < clearance + size:
            continue
        if any(math.hypot(*(c - o)) < 2.0 for o in centers):
            continue
        centers.append(c)
        segs += _box(c[0], c[1], size, size * rng.uniform(0.6, 1.4)) if rng.random() < 0.5 \
            else _cyl(c[0], c[1], size * 0.7)
    return np.asarray(segs, dtype=float)


def raycast(segs, pose, n=360, rmin=0.12, rmax=12.0, sigma=0.01, rng=None):
    """Ranges (inf = no return) for a full 360-degree scan from `pose` (x, y, yaw)."""
    ang = -math.pi + (2 * math.pi / n) * np.arange(n)
    th = pose[2] + ang
    d = np.column_stack([np.cos(th), np.sin(th)])
    p, e = segs[:, :2], segs[:, 2:] - segs[:, :2]
    w = p - pose[:2]
    denom = d[:, 0:1] * e[:, 1] - d[:, 1:2] * e[:, 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (w[:, 0] * e[:, 1] - w[:, 1] * e[:, 0]) / denom
        u = (w[:, 0:1].T * d[:, 1:2] - w[:, 1:2].T * d[:, 0:1]) / denom
    valid = (np.abs(denom) > 1e-9) & (t > 0) & (u >= 0) & (u <= 1)
    t = np.where(valid, t, np.inf).min(axis=1)
    r = np.where((t >= rmin) & (t <= rmax), t, np.inf)
    if rng is not None and sigma > 0:
        r = r + np.where(np.isfinite(r), rng.normal(0.0, sigma, n), 0.0)
    return r, -math.pi, 2 * math.pi / n


# ----------------------------------------------------------------------------- sensors
def simulate_imu(traj, rng, gyro_bias=0.012, accel_bias=(0.03, -0.02),
                 gyro_sigma=0.004, accel_sigma=0.05):
    """Body-frame IMU stream: gz, ax, ay (planar; gravity removed)."""
    gz = traj['w'] + gyro_bias + rng.normal(0.0, gyro_sigma, len(traj['w']))
    ax = traj['a_long'] + accel_bias[0] + rng.normal(0.0, accel_sigma, len(gz))
    ay = traj['v'] * traj['w'] + accel_bias[1] + rng.normal(0.0, accel_sigma, len(gz))
    return gz, ax, ay


def simulate_encoder(traj, rng, rate_div=2, v_scale=1.03, w_scale=0.93, w_bias=0.008,
                     v_sigma=0.01, w_sigma=0.01):
    """Wheel-odometry dead reckoning with scale error, yaw bias and noise (skid-steer slip).

    Returns poses at every `rate_div`-th trajectory sample, as (n, 3) in the start frame.
    """
    dt = 1.0 / traj['rate']
    n = len(traj['t'])
    v = traj['v'] * v_scale + rng.normal(0.0, v_sigma, n)
    w = traj['w'] * w_scale + w_bias * (traj['v'] > 0.05) + rng.normal(0.0, w_sigma, n)
    pose = np.zeros((n, 3))
    for i in range(1, n):
        ym = pose[i - 1, 2] + 0.5 * w[i - 1] * dt
        pose[i] = [pose[i - 1, 0] + v[i - 1] * math.cos(ym) * dt,
                   pose[i - 1, 1] + v[i - 1] * math.sin(ym) * dt,
                   wrap(pose[i - 1, 2] + w[i - 1] * dt)]
    return pose[::rate_div]


def lidar_points(segs, base_pose, lidar_to_base, rng):
    pose_l = compose(np.asarray(base_pose), np.asarray(lidar_to_base))
    r, amin, ainc = raycast(segs, pose_l, rng=rng)
    return scan_to_points(r, amin, ainc, 0.15, 10.0)


def ground_truth_in_start_frame(traj):
    g0 = np.array([traj['x'][0], traj['y'][0], traj['yaw'][0]])
    inv0 = inverse(g0)
    return np.array([compose(inv0, np.array([x, y, th]))
                     for x, y, th in zip(traj['x'], traj['y'], traj['yaw'])])


def make_corridor(duration=60.0, rate=100.0, speed=0.5, length=65.0, width=2.0):
    """Straight, featureless corridor: the classic degenerate case for scan matching.

    Returns (traj, segs). The back wall is visible only for the first few metres.
    """
    n = int(duration * rate)
    t = np.arange(n) / rate
    v = np.clip((t - 2.0) / 3.0, 0.0, 1.0) * speed
    x = np.cumsum(v) / rate
    traj = dict(t=t, x=x, y=np.zeros(n), yaw=np.zeros(n), v=v, w=np.zeros(n),
                a_long=np.gradient(v, 1.0 / rate), rate=rate)
    h = width / 2.0
    segs = np.array([(-5, -h, length - 5, -h), (-5, h, length - 5, h),
                     (-5, -h, -5, h), (length - 5, -h, length - 5, h)], dtype=float)
    return traj, segs
