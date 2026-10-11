"""Regression tests for the LIO core on synthetic worlds (no ROS needed): pytest test/"""
import math

import numpy as np

from state_estimation import synthetic_world as sw
from state_estimation.imu_core import ImuDeadReckoning
from state_estimation.lio_core import Lio2D, LioParams
from state_estimation.synthetic_eval import metrics, run

LIDAR = (0.10, 0.0, 0.0)


def test_lio_tracks_ground_truth_in_cluttered_arena():
    res = run(duration=40.0, seed=7)
    m = metrics(res['lio'], res['gt'])
    assert res['ok'].mean() > 0.99
    assert m['ate_rmse'] < 0.05 and m['yaw_rmse_deg'] < 0.5
    # and it must clearly beat dead reckoning from the IMU alone
    assert m['ate_rmse'] < 0.01 * metrics(res['imu'], res['gt'])['ate_rmse']


def test_gyro_bias_is_calibrated():
    res = run(duration=20.0, seed=3)
    assert res['lio_calib_ok'] and abs(res['gyro_bias'] - 0.012) < 0.002


def test_corridor_degeneracy_is_left_to_the_imu():
    """Along a featureless corridor the scan cannot observe the along-track direction.

    Expected behaviour: lateral error and yaw stay tight (the walls constrain them), and the
    along-track direction is left to the IMU prediction -- so LIO must be no worse than plain
    IMU dead reckoning there. (Both still drift along-track because of accelerometer bias; that
    is a physical limit, and is what the wheel encoders in the EKF are for.)
    """
    lio_err, imu_err = [], []
    for seed in range(6):
        rng = np.random.default_rng(seed)
        traj, segs = sw.make_corridor(duration=60.0)
        gz, ax, ay = sw.simulate_imu(traj, rng)
        lio, dr = Lio2D(LioParams(lidar_to_base=LIDAR)), ImuDeadReckoning()
        last = imu = None
        for i, t in enumerate(traj['t']):
            lio.add_imu(t, gz[i], ax[i], ay[i])
            imu = dr.add(t, gz[i], ax[i], ay[i]) or imu
            if i % 10 == 0:
                pts = sw.lidar_points(segs, [traj['x'][i], 0.0, 0.0], LIDAR, rng)
                last = lio.process_scan(t, pts) or last
        xt = traj['x'][5990]                       # truth at the last scan time
        assert not np.isnan(last.pose).any()
        assert abs(last.pose[1]) < 0.1 and abs(math.degrees(last.pose[2])) < 1.0
        lio_err.append(abs(last.pose[0] - xt))
        imu_err.append(abs(imu[0] - xt))
    assert np.mean(lio_err) <= np.mean(imu_err)


def test_first_scan_defines_origin():
    lio = Lio2D(LioParams())
    res = lio.process_scan(0.0, np.random.default_rng(0).uniform(-3, 3, (200, 2)))
    assert np.allclose(res.pose, 0.0) and res.ok
