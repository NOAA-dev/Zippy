"""IMU-only dead reckoning (numpy only, no ROS).

Gyro-z is integrated for yaw and body-frame accelerometer x/y is double-integrated for
position. Biases are taken from the first `init_duration` seconds (robot assumed stationary).
Position drifts quadratically with time -- that is expected, and it is exactly what makes this
useful as a baseline when comparing against the encoder, LIO and the fused EKF output.
"""
import math

import numpy as np


class ImuDeadReckoning:
    def __init__(self, init_duration=1.0):
        self.init_duration = init_duration
        self.x = self.y = self.yaw = 0.0
        self.vx_w = self.vy_w = 0.0            # world-frame velocity
        self.bg = 0.0
        self.ba = np.zeros(2)
        self.ready = False
        self._calib = []
        self._t0 = None
        self._tp = None
        self._prev = None                      # previous (gz, ax, ay)

    def add(self, t, gz, ax, ay):
        """Feed one IMU sample. Returns (x, y, yaw, vx_body, wz) or None while calibrating."""
        if not self.ready:
            if self._t0 is None:
                self._t0 = t
            self._calib.append((gz, ax, ay))
            if t - self._t0 >= self.init_duration:
                c = np.asarray(self._calib)
                self.bg = float(c[:, 0].mean())
                self.ba = c[:, 1:].mean(axis=0)
                self.ready = True
                self._tp, self._prev = t, (gz, ax, ay)
            return None
        dt = t - self._tp
        if dt <= 0.0:
            return None
        gz0, ax0, ay0 = self._prev
        w = 0.5 * (gz0 + gz) - self.bg
        a_b = 0.5 * np.array([ax0 + ax, ay0 + ay]) - self.ba
        yaw_mid = self.yaw + 0.5 * w * dt
        c, s = math.cos(yaw_mid), math.sin(yaw_mid)
        ax_w, ay_w = c * a_b[0] - s * a_b[1], s * a_b[0] + c * a_b[1]
        vx_new, vy_new = self.vx_w + ax_w * dt, self.vy_w + ay_w * dt
        self.x += 0.5 * (self.vx_w + vx_new) * dt
        self.y += 0.5 * (self.vy_w + vy_new) * dt
        self.vx_w, self.vy_w = vx_new, vy_new
        self.yaw = (self.yaw + w * dt + math.pi) % (2 * math.pi) - math.pi
        self._tp, self._prev = t, (gz, ax, ay)
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        vx_b = cy * self.vx_w + sy * self.vy_w
        return self.x, self.y, self.yaw, vx_b, w
