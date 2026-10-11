#!/usr/bin/env python3
"""Plot an odom_viz CSV log: every odometry source on one set of axes.

    ros2 run state_estimation plot_odometry                      # newest log in ~/zippy_odom_logs
    ros2 run state_estimation plot_odometry run.csv --out run.png --ref ekf

Panels: (1) x-y trajectories, (2) yaw vs time, (3) distance of each source from the reference
source (default: ekf if present, else lio) -- how much the sources disagree over time.
There is no ground truth on the robot, so the EKF output is the best available reference;
'truth' (if present in the log) is drawn in black.
"""
import argparse
import csv
import glob
import os

import numpy as np

COLORS = {'encoder': '#ff8c00', 'imu': '#dc28dc', 'lio': '#00c8ff', 'ekf': '#28dc3c', 'truth': 'k'}
ORDER = ['truth', 'encoder', 'imu', 'lio', 'ekf']


def load(path):
    rows = {}
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            rows.setdefault(r['source'], []).append((float(r['t']), float(r['x']), float(r['y']), float(r['yaw'])))
    out = {}
    for k, v in rows.items():
        a = np.array(v)
        a = a[np.argsort(a[:, 0])]
        a[:, 0] -= 0.0
        out[k] = a
    if not out:
        raise SystemExit(f'{path}: no rows')
    t0 = min(a[0, 0] for a in out.values())
    for a in out.values():
        a[:, 0] -= t0
    return out


def summary(data, ref):
    print(f"{'source':10s} {'path m':>8s} {'end x':>8s} {'end y':>8s} {'end yaw deg':>12s} {'end dist to ' + ref + ' m':>20s}")
    for k in sorted(data, key=lambda s: ORDER.index(s) if s in ORDER else 99):
        a = data[k]
        plen = float(np.sum(np.hypot(*np.diff(a[:, 1:3], axis=0).T)))
        d = np.hypot(a[-1, 1] - data[ref][-1, 1], a[-1, 2] - data[ref][-1, 2]) if ref in data else float('nan')
        print(f'{k:10s} {plen:8.2f} {a[-1, 1]:8.2f} {a[-1, 2]:8.2f} {np.degrees(a[-1, 3]):12.1f} {d:20.2f}')


def plot(data, ref, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = sorted(data, key=lambda s: ORDER.index(s) if s in ORDER else 99)
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1])
    a0, a1, a2 = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    for k in names:
        a = data[k]
        c = COLORS.get(k)
        lw = 2.6 if k in ('ekf', 'truth') else 1.4
        a0.plot(a[:, 1], a[:, 2], color=c, lw=lw, label=k, alpha=0.5 if k == 'truth' else 1.0)
        a0.plot(a[-1, 1], a[-1, 2], 'o', color=c, ms=5)
        a1.plot(a[:, 0], np.degrees(np.unwrap(a[:, 3])), color=c, lw=lw, alpha=0.5 if k == 'truth' else 1.0)
    a0.plot(0, 0, 'k*', ms=12, label='start')
    a0.set_aspect('equal')
    a0.set_title('Trajectories (all sources start at the origin = start pose)')
    a0.set_xlabel('x [m]')
    a0.set_ylabel('y [m]')
    a0.legend()
    a0.grid(alpha=0.3)
    a1.set_title('Yaw')
    a1.set_ylabel('yaw [deg]')
    a1.grid(alpha=0.3)
    if ref in data:
        r = data[ref]
        for k in names:
            if k == ref:
                continue
            a = data[k]
            x = np.interp(r[:, 0], a[:, 0], a[:, 1])
            y = np.interp(r[:, 0], a[:, 0], a[:, 2])
            a2.plot(r[:, 0], np.hypot(x - r[:, 1], y - r[:, 2]), color=COLORS.get(k), label=k)
        a2.set_yscale('symlog', linthresh=0.01)
        a2.legend()
    a2.set_title(f'Distance from {ref}')
    a2.set_xlabel('time [s]')
    a2.set_ylabel('[m]')
    a2.grid(alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f'saved {out}')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('csv', nargs='?', default='')
    ap.add_argument('--out', default='')
    ap.add_argument('--ref', default='')
    a = ap.parse_args(argv)
    path = a.csv
    if not path:
        logs = sorted(glob.glob(os.path.expanduser('~/zippy_odom_logs/odom_*.csv')))
        if not logs:
            raise SystemExit('no CSV given and none found in ~/zippy_odom_logs')
        path = logs[-1]
    data = load(path)
    ref = a.ref or ('ekf' if 'ekf' in data else 'lio' if 'lio' in data else next(iter(data)))
    summary(data, ref)
    plot(data, ref, a.out or os.path.splitext(path)[0] + '.png')


if __name__ == '__main__':
    main()
