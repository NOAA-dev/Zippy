"""Run LIO, IMU-only and encoder odometry on a synthetic world and score them against truth.

    ros2 run state_estimation lio_synthetic_eval --out lio_eval.png
or  python3 -m state_estimation.synthetic_eval --duration 150 --seed 1
"""
import argparse
import time

import numpy as np

from .imu_core import ImuDeadReckoning
from .lio_core import Lio2D, LioParams, compose, inverse, wrap
from . import synthetic_world as sw


def run(duration=150.0, seed=0, lidar_to_base=(0.10, 0.0, 0.0), params=None):
    rng = np.random.default_rng(seed)
    traj = sw.make_trajectory(duration)
    segs = sw.make_world(traj, seed=seed)
    gz, ax, ay = sw.simulate_imu(traj, rng)
    enc = sw.simulate_encoder(traj, rng)                  # every 2nd sample (50 Hz)
    gt = sw.ground_truth_in_start_frame(traj)
    prm = params or LioParams(lidar_to_base=lidar_to_base)
    lio = Lio2D(prm)
    imu_dr = ImuDeadReckoning()
    t_arr = traj['t']

    scan_t, lio_pose, lio_ok, gt_scan, imu_pose, enc_pose = [], [], [], [], [], []
    last_imu = np.zeros(3)
    t0 = time.perf_counter()
    for i, t in enumerate(t_arr):
        lio.add_imu(t, gz[i], ax[i], ay[i])
        r = imu_dr.add(t, gz[i], ax[i], ay[i])
        if r is not None:
            last_imu = np.array(r[:3])
        if i % 10 == 0:                                   # 10 Hz scan
            base = np.array([traj['x'][i], traj['y'][i], traj['yaw'][i]])
            pts = sw.lidar_points(segs, base, prm.lidar_to_base, rng)
            res = lio.process_scan(t, pts)
            if res is not None:
                scan_t.append(t)
                lio_pose.append(res.pose)
                lio_ok.append(res.ok)
                gt_scan.append(gt[i])
                imu_pose.append(last_imu.copy())
                enc_pose.append(enc[i // 2])
    wall = time.perf_counter() - t0
    out = dict(t=np.array(scan_t), lio=np.array(lio_pose), ok=np.array(lio_ok),
               gt=np.array(gt_scan), imu=np.array(imu_pose), enc=np.array(enc_pose),
               segs=segs, gt_full=gt, lio_calib_ok=lio.calib_ok, gyro_bias=lio.bg,
               ms_per_scan=1000.0 * wall / max(len(scan_t), 1))
    return out


def metrics(est, gt):
    e = est[:, :2] - gt[:, :2]
    pos = np.hypot(e[:, 0], e[:, 1])
    yaw = np.abs(np.array([wrap(a) for a in est[:, 2] - gt[:, 2]]))
    path = np.sum(np.hypot(*np.diff(gt[:, :2], axis=0).T))
    return dict(ate_rmse=float(np.sqrt(np.mean(pos ** 2))), final=float(pos[-1]),
                final_pct=float(100.0 * pos[-1] / path), yaw_rmse_deg=float(np.degrees(np.sqrt(np.mean(yaw ** 2)))),
                yaw_final_deg=float(np.degrees(yaw[-1])))


def report(res):
    rows = [('LIO', res['lio']), ('IMU only', res['imu']), ('Encoder', res['enc'])]
    path = np.sum(np.hypot(*np.diff(res['gt'][:, :2], axis=0).T))
    print(f"path length {path:.1f} m, {len(res['t'])} scans, "
          f"{res['ms_per_scan']:.1f} ms/scan, scan-match accepted {100 * res['ok'].mean():.1f}%, "
          f"IMU calibration ok={res['lio_calib_ok']}, final gyro bias {res['gyro_bias']:.4f} rad/s (true 0.0120)")
    print(f"{'source':10s} {'ATE rmse m':>11s} {'final err m':>12s} {'final %path':>12s} {'yaw rmse deg':>13s} {'final yaw deg':>14s}")
    out = {}
    for name, est in rows:
        m = metrics(est, res['gt'])
        out[name] = m
        print(f"{name:10s} {m['ate_rmse']:11.3f} {m['final']:12.3f} {m['final_pct']:12.2f} "
              f"{m['yaw_rmse_deg']:13.2f} {m['yaw_final_deg']:14.2f}")
    return out


def plot(res, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1.15, 1]})
    a = ax[0]
    a.set_aspect('equal')
    a.set_title('Trajectories in the start frame (all sources begin at the origin)')
    for name, est, c, lw in (('ground truth', res['gt'], 'k', 4.0), ('encoder', res['enc'], 'tab:orange', 1.6),
                             ('IMU only', res['imu'], 'tab:purple', 1.6), ('LIO', res['lio'], 'tab:cyan', 1.4)):
        a.plot(est[:, 0], est[:, 1], '-', color=c, lw=lw, label=name, alpha=0.45 if name == 'ground truth' else 1.0)
    a.set_xlabel('x [m]')
    a.set_ylabel('y [m]')
    a.legend()
    a.grid(alpha=0.3)
    b = ax[1]
    for name, est, c in (('encoder', res['enc'], 'tab:orange'), ('IMU only', res['imu'], 'tab:purple'),
                         ('LIO', res['lio'], 'tab:cyan')):
        b.plot(res['t'], np.hypot(*(est[:, :2] - res['gt'][:, :2]).T), color=c, label=name)
    b.set_yscale('log')
    b.set_xlabel('time [s]')
    b.set_ylabel('position error vs ground truth [m]')
    b.set_title('Drift')
    b.legend()
    b.grid(alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f'saved {path}')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--duration', type=float, default=150.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='')
    a = ap.parse_args(argv)
    res = run(a.duration, a.seed)
    report(res)
    if a.out:
        plot(res, a.out)


if __name__ == '__main__':
    main()
