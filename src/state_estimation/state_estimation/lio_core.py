"""2D LiDAR-inertial odometry (LIO) core. numpy/scipy only, no ROS imports.

Per LiDAR scan:
  1. IMU propagation: integrate bias-compensated gyro-z and body-frame accelerometer
     between the previous and current scan stamps -> predicted pose.
  2. Scan-to-local-submap matching: Gauss-Newton point-to-line ICP on (x, y, yaw) with the
     IMU prediction as a Gaussian prior (MAP estimate). Directions the scan cannot observe
     (e.g. along a straight corridor) therefore stay at the IMU prediction instead of drifting.
  3. Velocity update, online gyro-bias update, keyframe insertion.

The submap is a sliding window of the last few keyframes: there is NO global map and NO loop
closure, so this is odometry, not SLAM. The wheel encoders are deliberately not used, which
keeps this an independent source for the EKF.

Conventions: poses are (x, y, yaw) in the odom frame (origin = start pose); velocities are
expressed in the robot (base) frame. The IMU is assumed aligned with the base frame.
"""
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree


# ----------------------------------------------------------------------------- SE(2) helpers
def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def rot(th):
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s], [s, c]])


def compose(a, b):
    """a (+) b : pose b expressed in frame a, returned in a's parent frame."""
    c, s = math.cos(a[2]), math.sin(a[2])
    return np.array([a[0] + c * b[0] - s * b[1],
                     a[1] + s * b[0] + c * b[1],
                     wrap(a[2] + b[2])])


def inverse(a):
    c, s = math.cos(a[2]), math.sin(a[2])
    return np.array([-(c * a[0] + s * a[1]), s * a[0] - c * a[1], wrap(-a[2])])


def transform(pose, pts):
    c, s = math.cos(pose[2]), math.sin(pose[2])
    out = np.empty_like(pts)
    out[:, 0] = c * pts[:, 0] - s * pts[:, 1] + pose[0]
    out[:, 1] = s * pts[:, 0] + c * pts[:, 1] + pose[1]
    return out


def voxel_downsample(pts, size):
    if len(pts) == 0 or size <= 0:
        return pts
    keys = np.floor(pts / size).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[np.sort(idx)]


def scan_to_points(ranges, angle_min, angle_inc, range_min, range_max):
    """LaserScan ranges -> (N, 2) points in the lidar frame (invalid returns dropped)."""
    r = np.asarray(ranges, dtype=float)
    ang = angle_min + angle_inc * np.arange(len(r))
    ok = np.isfinite(r) & (r >= range_min) & (r <= range_max)
    return np.column_stack([r[ok] * np.cos(ang[ok]), r[ok] * np.sin(ang[ok])])


# ----------------------------------------------------------------------------- parameters
@dataclass
class LioParams:
    lidar_to_base: tuple = (0.0, 0.0, 0.0)  # lidar pose in the base frame: x, y, yaw
    range_min: float = 0.15
    range_max: float = 10.0          # ignore far returns (beam-spread / low-accuracy region)
    src_voxel: float = 0.05          # downsample of the incoming scan [m]
    map_voxel: float = 0.08          # downsample of the submap [m]
    keyframe_dist: float = 0.30      # new keyframe after this much translation [m]
    keyframe_angle: float = 0.17     # ... or this much rotation [rad]
    submap_keyframes: int = 12       # sliding-window size
    normal_k: int = 6                # neighbours for submap line normals
    normal_max_ratio: float = 0.10   # lambda_min/lambda_max above this -> not a line (corner)
    max_iters: int = 20
    gate_start: float = 0.6          # correspondence gate, shrinks to gate_end [m]
    gate_end: float = 0.2
    sigma_range: float = 0.03        # point-to-line residual sigma [m]
    huber: float = 0.05              # robust-loss knee [m]
    min_inliers: int = 40
    min_inlier_ratio: float = 0.25
    # IMU-prediction prior, per 0.1 s scan interval (scaled up for longer gaps)
    prior_sigma_long: float = 0.04   # along heading [m]
    prior_sigma_lat: float = 0.03    # across heading [m]
    prior_sigma_yaw: float = 0.01    # [rad]
    max_corr_xy: float = 0.5         # reject a match that moves this far from the prior [m]
    max_corr_yaw: float = 0.4        # [rad]
    use_accel: bool = True
    init_duration: float = 1.0       # robot assumed stationary this long to calibrate IMU biases
    gyro_bias_alpha: float = 0.005   # online gyro-bias low-pass gain per scan
    cov_inflation: float = 3.0       # scale on the reported per-step covariance
    degeneracy_eig_min: float = 3e3  # scan information below this [1/m^2] is treated as unobservable
    yaw_scale_m: float = 3.0         # length used to make yaw comparable to x, y in the eigen test


@dataclass
class LioResult:
    t: float
    pose: np.ndarray                 # x, y, yaw in odom
    vel: np.ndarray                  # vx, vy [m/s], wz [rad/s] in base frame
    cov: np.ndarray                  # 3x3 covariance of the per-step (x, y, yaw) estimate
    ok: bool                         # False -> scan match rejected, IMU prediction only
    inlier_ratio: float = 0.0
    gyro_bias: float = 0.0
    info: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- the estimator
class Lio2D:
    def __init__(self, params=None):
        self.p = params or LioParams()
        self.pose = np.zeros(3)
        self.vel = np.zeros(2)            # body-frame velocity at the last scan
        self.bg = 0.0                     # gyro-z bias [rad/s]
        self.ba = np.zeros(2)             # accel bias (body x, y) [m/s^2]
        self.calib_ok = None              # None = still calibrating
        self._imu = deque(maxlen=4000)
        self._last_imu = (0.0, 0.0, 0.0)  # gz, ax, ay held between samples
        self._have_imu = False
        self._calib = []
        self._t0_imu = None
        self._t_prev = None
        self._kfs = deque(maxlen=self.p.submap_keyframes)
        self._last_kf_pose = None
        self._map = None                  # (tree, points, normals) in odom frame
        self._lam_min = float('inf')      # weakest scan-constraint eigenvalue (diagnostic)
        self._T_bl = np.array(self.p.lidar_to_base, dtype=float)

    def set_lidar_extrinsic(self, x, y, yaw):
        """Pose of the lidar in the base frame (e.g. from TF)."""
        self._T_bl = np.array([x, y, yaw], dtype=float)

    # ------------------------------------------------------------------ IMU input
    def add_imu(self, t, gz, ax, ay):
        self._imu.append((t, gz, ax, ay))
        if self.calib_ok is not None:
            return
        if self._t0_imu is None:
            self._t0_imu = t
        self._calib.append((gz, ax, ay))
        if t - self._t0_imu >= self.p.init_duration:
            c = np.asarray(self._calib)
            still = abs(c[:, 0].mean()) < 0.05 and c[:, 0].std() < 0.05
            if still:
                self.bg = float(c[:, 0].mean())
                self.ba = c[:, 1:].mean(axis=0)
            self.calib_ok = bool(still)   # False -> robot was moving; biases left at zero
            self._calib = []

    def _propagate(self, t):
        """Integrate buffered IMU over (t_prev, t].

        Returns (d_pose, v_end, dth_raw): pose increment and end velocity, both in the
        previous body frame, and the raw (bias-uncorrected) gyro yaw increment.
        """
        tc = self._t_prev
        dth, dth_raw = 0.0, 0.0
        v = self.vel.copy()
        d = np.zeros(2)
        gz0, ax0, ay0 = self._last_imu
        samples = []
        while self._imu and self._imu[0][0] <= t:
            s = self._imu.popleft()
            if s[0] > tc:
                samples.append(s)
        samples.append((t, gz0, ax0, ay0))       # close the interval at the scan stamp
        for ts, g, axx, ayy in samples:
            dt = ts - tc
            if dt <= 0.0:
                continue
            gm = 0.5 * (gz0 + g)
            w = gm - self.bg
            if self.p.use_accel and self._have_imu:
                a_body = 0.5 * np.array([ax0 + axx, ay0 + ayy]) - self.ba
                a0 = rot(dth + 0.5 * w * dt) @ a_body     # accel in the previous body frame
            else:
                a0 = np.zeros(2)
            v_new = v + a0 * dt
            d += 0.5 * (v + v_new) * dt
            v = v_new
            dth += w * dt
            dth_raw += gm * dt
            tc = ts
            gz0, ax0, ay0 = g, axx, ayy
        if len(samples) > 1:
            self._have_imu = True
            self._last_imu = (gz0, ax0, ay0)
        return np.array([d[0], d[1], dth]), v, dth_raw

    # ------------------------------------------------------------------ submap
    def _build_map(self):
        k = self.p.normal_k
        pts = voxel_downsample(np.vstack(self._kfs), self.p.map_voxel)
        if len(pts) <= k:
            self._map = None
            return
        dist, nbr = cKDTree(pts).query(pts, k=k)
        nb = pts[nbr]                                          # (M, k, 2)
        dd = nb - nb.mean(axis=1, keepdims=True)
        cov = np.einsum('mki,mkj->mij', dd, dd) / k
        w, v = np.linalg.eigh(cov)                             # ascending eigenvalues
        normals = v[:, :, 0]
        good = (w[:, 0] <= self.p.normal_max_ratio * w[:, 1] + 1e-9) & (dist[:, -1] < 0.4)
        if good.sum() <= self.p.min_inliers:
            self._map = None
            return
        pts, normals = pts[good], normals[good]
        self._map = (cKDTree(pts), pts, normals)

    def _add_keyframe(self, pts_base):
        self._kfs.append(transform(self.pose, pts_base))
        self._last_kf_pose = self.pose.copy()
        self._build_map()

    def _need_keyframe(self):
        rel = compose(inverse(self._last_kf_pose), self.pose)
        return math.hypot(rel[0], rel[1]) > self.p.keyframe_dist or abs(rel[2]) > self.p.keyframe_angle

    # ------------------------------------------------------------------ scan matching
    def _match(self, src, prior_pose, P):
        """MAP scan-to-submap alignment. Returns (pose, H, n_inliers) or None."""
        tree, tgt, nrm = self._map
        p = self.p
        pose = prior_pose.copy()
        H = None
        n_in = 0
        inv_s2 = 1.0 / p.sigma_range ** 2
        for it in range(p.max_iters):
            frac = min(1.0, it / (0.5 * p.max_iters))
            gate = p.gate_start + (p.gate_end - p.gate_start) * frac
            q = transform(pose, src)
            dist, idx = tree.query(q, distance_upper_bound=gate)
            ok = np.isfinite(dist)
            n_in = int(ok.sum())
            if n_in < p.min_inliers:
                return None
            nv = nrm[idx[ok]]
            r = np.einsum('ij,ij->i', nv, q[ok] - tgt[idx[ok]])
            ar = np.maximum(np.abs(r), 1e-12)
            w = np.where(ar <= p.huber, 1.0, p.huber / ar)
            c, s = math.cos(pose[2]), math.sin(pose[2])
            ps = src[ok]
            dq = np.column_stack([-s * ps[:, 0] - c * ps[:, 1], c * ps[:, 0] - s * ps[:, 1]])
            J = np.column_stack([nv, np.einsum('ij,ij->i', nv, dq)])
            Hd = (J.T * (w * inv_s2)) @ J                      # scan information
            gd = J.T @ (w * r) * inv_s2
            # Degeneracy handling: keep the scan's contribution only along directions it really
            # constrains (eigen-test in x, y, yaw*L units); the IMU prior decides the rest.
            S = np.array([1.0, 1.0, p.yaw_scale_m])
            lam, V = np.linalg.eigh(Hd / np.outer(S, S))
            keep = lam > p.degeneracy_eig_min
            self._lam_min = float(lam[0])
            Vk = V[:, keep]
            Hd = (Vk * lam[keep]) @ Vk.T * np.outer(S, S)
            gd = S * (Vk @ (Vk.T @ (gd / S)))
            H = Hd + P
            err = pose - prior_pose
            err[2] = wrap(err[2])
            g = gd + P @ err
            delta = -np.linalg.solve(H, g)
            pose = pose + delta
            pose[2] = wrap(pose[2])
            if np.abs(delta).max() < 1e-4:
                break
        return pose, H, n_in

    def _prior_info(self, prior_pose, dt):
        p = self.p
        s = max(1.0, dt / 0.1)
        Pb = np.diag([1.0 / (p.prior_sigma_long * s) ** 2,
                      1.0 / (p.prior_sigma_lat * s) ** 2])
        # the prior is expressed along the heading at the START of the step
        R0 = rot(self.pose[2])
        P = np.zeros((3, 3))
        P[:2, :2] = R0 @ Pb @ R0.T
        P[2, 2] = 1.0 / (p.prior_sigma_yaw * s) ** 2
        return P

    # ------------------------------------------------------------------ main entry
    def process_scan(self, t, pts_lidar):
        """Feed one scan (points in the lidar frame). Returns LioResult, or None if stale."""
        p = self.p
        pts = voxel_downsample(transform(self._T_bl, pts_lidar), p.src_voxel)

        if self._t_prev is None:                       # first scan: origin + first keyframe
            self._t_prev = t
            self._add_keyframe(pts)
            return LioResult(t, self.pose.copy(), np.zeros(3), np.eye(3) * 1e-6, True, 1.0, self.bg)

        dt = t - self._t_prev
        if dt <= 0.0:
            return None

        d_pose, v_end, dth_raw = self._propagate(t)
        prior_pose = compose(self.pose, d_pose)
        P = self._prior_info(prior_pose, dt)

        ok, inl_ratio, cov = False, 0.0, None
        new_pose = prior_pose
        if self._map is not None and len(pts) >= p.min_inliers:
            res = self._match(pts, prior_pose, P)
            if res is not None:
                pose_m, H, n_in = res
                corr = pose_m - prior_pose
                corr[2] = wrap(corr[2])
                inl_ratio = n_in / len(pts)
                if (inl_ratio >= p.min_inlier_ratio and math.hypot(corr[0], corr[1]) <= p.max_corr_xy
                        and abs(corr[2]) <= p.max_corr_yaw):
                    ok, new_pose = True, pose_m
                    cov = np.linalg.inv(H) * p.cov_inflation
        if not ok:                                     # fall back to IMU prediction only
            cov = np.linalg.inv(P) * 10.0

        # velocity: ICP-refined average velocity, corrected to end-of-interval with IMU accel
        delta = compose(inverse(self.pose), new_pose)  # step in the previous body frame
        v_avg = delta[:2] / dt
        v_end_icp = v_avg + 0.5 * (v_end - self.vel) if p.use_accel else v_avg
        wz = delta[2] / dt

        # slow online gyro-bias estimate, only when the scan constrains yaw well
        if ok and self.calib_ok is not None and cov[2, 2] < (0.01) ** 2:
            b_meas = (dth_raw - delta[2]) / dt
            self.bg = (1.0 - p.gyro_bias_alpha) * self.bg + p.gyro_bias_alpha * b_meas

        self.vel = rot(-delta[2]) @ v_end_icp
        self.pose = new_pose
        self._t_prev = t
        if ok and self._need_keyframe():
            self._add_keyframe(pts)

        return LioResult(t, self.pose.copy(), np.array([self.vel[0], self.vel[1], wz]), cov, ok,
                         inl_ratio, self.bg, {'lam_min': self._lam_min})
