import numpy as np
from scipy.spatial.transform import Rotation as R

from quadrotor import Quadrotor3D
from utils import transform_trajectory, v_dot_q, quaternion_inverse, quaternion_to_euler


# ============================================================
# Utility helpers
# ============================================================

def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


def norm_safe(v, eps=1e-9):
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v), eps
    return v / n, n


def quat_normalize(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_conjugate(q):
    # q = [w,x,y,z]
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quat_multiply(q1, q2):
    # Hamilton product, q = [w,x,y,z]
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=float)


def quat_inverse(q):
    return quat_conjugate(q) / np.dot(q, q)


def quat_error(q_des, q):
    """
    Quaternion attitude error:
        q_err = q_des * q^{-1}
    Both in [w,x,y,z]
    """
    q_err = quat_multiply(q_des, quat_inverse(q))
    if q_err[0] < 0.0:
        q_err = -q_err
    return quat_normalize(q_err)


def quat_wxyz_to_xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]], dtype=float)


def quat_xyzw_to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]], dtype=float)

def quat_to_rotmat(q_wxyz):
    return R.from_quat(quat_wxyz_to_xyzw(q_wxyz)).as_matrix()


def rotmat_to_quat_wxyz(Rm):
    return quat_xyzw_to_wxyz(R.from_matrix(Rm).as_quat())


def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def lowpass(prev, new, alpha):
    return alpha * new + (1.0 - alpha) * prev


# ============================================================
# PID blocks with anti-windup + derivative-on-measurement
# ============================================================

class PIDAxis:
    def __init__(self, kp=0.0, ki=0.0, kd=0.0,
                 integral_limit=np.inf,
                 output_limit=np.inf,
                 d_lpf_alpha=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.integral_limit = integral_limit
        self.output_limit = output_limit
        self.d_lpf_alpha = d_lpf_alpha

        self.integral = 0.0
        self.prev_measurement = None
        self.prev_error = None
        self.d_state = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_measurement = None
        self.prev_error = None
        self.d_state = 0.0

    def update(self, error, dt,
               measurement=None,
               feedforward=0.0,
               freeze_integrator=False,
               output_limit_override=None):
        """
        derivative-on-measurement:
            d_term = -kd * d(measurement)/dt
        """
        if dt <= 1e-6:
            return feedforward

        # Derivative on measurement
        d_meas = 0.0
        if measurement is not None and self.prev_measurement is not None:
            raw_d = (measurement - self.prev_measurement) / dt
            self.d_state = lowpass(self.d_state, raw_d, self.d_lpf_alpha)
            d_meas = self.d_state

        # Integrator
        if not freeze_integrator:
            self.integral += error * dt
            self.integral = clamp(self.integral, -self.integral_limit, self.integral_limit)

        u = (
            self.kp * error +
            self.ki * self.integral -
            self.kd * d_meas +
            feedforward
        )

        lim = self.output_limit if output_limit_override is None else output_limit_override
        u_sat = clamp(u, -lim, lim)

        # Simple anti-windup backoff
        if abs(u - u_sat) > 1e-9 and self.ki > 1e-9:
            self.integral *= 0.98

        self.prev_measurement = measurement
        self.prev_error = error

        return u_sat


class PIDVec3:
    def __init__(self, kp, ki, kd,
                 integral_limit=np.inf,
                 output_limit=np.inf,
                 d_lpf_alpha=1.0):
        self.axes = [
            PIDAxis(kp[0], ki[0], kd[0], integral_limit, output_limit, d_lpf_alpha),
            PIDAxis(kp[1], ki[1], kd[1], integral_limit, output_limit, d_lpf_alpha),
            PIDAxis(kp[2], ki[2], kd[2], integral_limit, output_limit, d_lpf_alpha),
        ]

    def reset(self):
        for a in self.axes:
            a.reset()

    def update(self, error, dt,
               measurement=None,
               feedforward=None,
               freeze_integrator=False,
               output_limit_override=None):
        out = np.zeros(3)
        for i in range(3):
            meas_i = None if measurement is None else measurement[i]
            ff_i = 0.0 if feedforward is None else feedforward[i]
            out[i] = self.axes[i].update(
                error=error[i],
                dt=dt,
                measurement=meas_i,
                feedforward=ff_i,
                freeze_integrator=freeze_integrator,
                output_limit_override=output_limit_override
            )
        return out


# ============================================================
# Main Controller
# ============================================================

class PIDController:
    def __init__(self, quad: Quadrotor3D, dt,
                 q_cost=None, r_cost=None, q_mask=None, rdrv_d_mat=None,
                 model_name="quad_3d_pid", solver_options=None):
        """
        Cascaded PID controller with similar interface to your MPC Controller.

        State convention:
            x = [p_xyz, q_wxyz, v_xyz, r_xyz]
        Control convention:
            u = [u1, u2, u3, u4], normalized such that thrust_i = u_i * quad.max_thrust
        """

        self.quad = quad
        self.dt = dt

        self.max_u = quad.max_input_value
        self.min_u = quad.min_input_value

        self.mass = quad.mass
        self.g = 9.81
        self.J = np.array(quad.J, dtype=float)

        self.x_f = np.array(quad.x_f, dtype=float).reshape(4)
        self.y_f = np.array(quad.y_f, dtype=float).reshape(4)
        self.c_f = np.array(quad.z_l_tau, dtype=float).reshape(4)

        self.max_motor_thrust = quad.max_thrust
        self.max_total_thrust = 4.0 * self.max_motor_thrust

        self.rdrv_d_mat = rdrv_d_mat

        # ------------------------------------------------------------
        # Internal state
        # ------------------------------------------------------------
        self.time_traj = None
        self.last_closest_index = 0
        self.desired_yaw = 0.0
        self.prev_u = np.ones(4) * ((self.mass * self.g / 4.0) / self.max_motor_thrust)

        # ------------------------------------------------------------
        # Tuning / limits
        # ------------------------------------------------------------
        self.max_vel_cmd = np.array([1.0, 1.0, 1.0])
        self.max_acc_cmd = np.array([2.0, 2.0, 2.0])
        self.max_tilt = np.deg2rad(20.0)
        self.max_rate_cmd = np.array([4.0, 4.0, 4.0])  # rad/s
        self.max_torque_cmd = np.array([1.5, 1.5, 1.5])  # N*m-ish scale
        self.motor_smoothing_alpha = 0.6

        # ------------------------------------------------------------
        # PID stacks 
        # ------------------------------------------------------------

        # Position -> desired velocity
        self.pos_pid = PIDVec3(
            kp=np.array([0.2, 0.1, 0.3]),
            ki=np.array([0.02, 0.1, 0.02]),  # reduced
            kd=np.array([0.1, 0.1, 0.5]),
            integral_limit=2.0,
            output_limit=5.0,
            d_lpf_alpha=0.2
        )

        # Velocity -> desired acceleration
        self.vel_pid = PIDVec3(
            kp=np.array([0.4, 0.3, 0.4]),
            ki=np.array([0.01, 0.17, 0.01]),
            kd=np.array([0.6, 0.4, 0.6]),
            integral_limit=2.0,
            output_limit=10.0,
            d_lpf_alpha=0.2
        )

        # Attitude P (quaternion error -> desired rates)
        self.att_kp = np.array([6, 6, 6])

        # Rate PID (body rate -> torques)
        self.rate_pid = PIDVec3(
            kp=np.array([1.7, 1.7, 1.7]),
            ki=np.array([0.01, 0.01, 0.01]),
            kd=np.array([0.1, 0.1, 0.1]),
            integral_limit=1.0,
            output_limit=2.0,
            d_lpf_alpha=0.2
        )

        # Optional user overrides
        if solver_options is not None:
            gains = solver_options.get("gains", {})
            self.set_gains(gains)

            if "max_vel_cmd" in solver_options:
                self.max_vel_cmd = np.array(solver_options["max_vel_cmd"], dtype=float)
            if "max_acc_cmd" in solver_options:
                self.max_acc_cmd = np.array(solver_options["max_acc_cmd"], dtype=float)
            if "max_tilt_deg" in solver_options:
                self.max_tilt = np.deg2rad(float(solver_options["max_tilt_deg"]))
            if "max_rate_cmd" in solver_options:
                self.max_rate_cmd = np.array(solver_options["max_rate_cmd"], dtype=float)
            if "motor_smoothing_alpha" in solver_options:
                self.motor_smoothing_alpha = float(solver_options["motor_smoothing_alpha"])

    # ============================================================
    # Public API
    # ============================================================

    def set_gains(self, gains: dict):
        """
        Optional tuning override helper.
        Example:
            gains = {
                "att_kp": [6,6,3],
                "pos_kp": [1,1,2]
            }
        """
        if "att_kp" in gains:
            self.att_kp = np.array(gains["att_kp"], dtype=float)

    def reset(self):
        self.pos_pid.reset()
        self.vel_pid.reset()
        self.rate_pid.reset()
        self.last_closest_index = 0
        self.prev_u = np.ones(4) * ((self.mass * self.g / 4.0) / self.max_motor_thrust)

    def update_trajectory(self, trajectory, preferred_speed=None):
        """
        Sets trajectory for the controlled object to pursue

        :param trajectory: 2d numpy array of (N+1 or more) elements with size 3 [x,y,z]
        """
        trajectory = np.asarray(trajectory, dtype=float)

        self.time_traj = trajectory

        self.last_closest_index = 0

    def run_optimization(self, initial_state=None):
        """
        Same user-facing role as your MPC method:
            compute next 4 motor commands.

        :param initial_state: 13-element list/array [p(3), q(4), v(3), w(3)]
        :return: np.array shape (4,)
        """

        if initial_state is None:
            initial_state = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        else:
            initial_state = np.asarray(initial_state, dtype=float)

        p = initial_state[0:3]
        q = quat_normalize(initial_state[3:7])   # [w,x,y,z]
        v = initial_state[7:10]
        w = initial_state[10:13]                 # body rates

        if self.time_traj is None or len(self.time_traj) == 0:
            self.time_traj = np.array([p.copy()])

        # ------------------------------------------------------------
        # 1) Local trajectory extraction (similar spirit to your MPC)
        # ------------------------------------------------------------
        starting_index = np.argmin(
            np.sum((self.time_traj[self.last_closest_index:] - p) ** 2, axis=1)
        ) + self.last_closest_index

        local_trajectory = self.time_traj[starting_index:]
        self.last_closest_index = starting_index

        # Lookahead target for smoother path following
        lookahead_idx = min(3, len(local_trajectory) - 1)
        p_ref = local_trajectory[lookahead_idx]

        # Feedforward from local path
        v_ff, a_ff = self.compute_feedforward(local_trajectory)

        # Optional yaw from path tangent
        path_dir = local_trajectory[min(2, len(local_trajectory) - 1)] - local_trajectory[0]
        if np.linalg.norm(path_dir[:2]) > 1e-3:
            self.desired_yaw = np.arctan2(path_dir[1], path_dir[0])

        # ------------------------------------------------------------
        # 2) Position loop: p error -> desired velocity
        # ------------------------------------------------------------
        pos_error = p_ref - p

        v_pid = self.pos_pid.update(
            error=pos_error,
            dt=self.dt,
            measurement=p,
            feedforward=v_ff
        )
        v_ref = clamp(v_pid, -self.max_vel_cmd, self.max_vel_cmd)

        # ------------------------------------------------------------
        # 3) Velocity loop: v error -> desired acceleration
        # ------------------------------------------------------------
        vel_error = v_ref - v

        # Optional linear drag compensation in world frame
        drag_comp = np.zeros(3)
        if self.rdrv_d_mat is not None:
            # velocity in body frame
            v_b = np.array(v_dot_q(v, quaternion_inverse(q))).astype(float).reshape(-1)
            drag_b = self.rdrv_d_mat @ v_b
            drag_comp = np.array(v_dot_q(drag_b, q)).astype(float).reshape(-1)

        a_pid = self.vel_pid.update(
            error=vel_error,
            dt=self.dt,
            measurement=v,
            feedforward=a_ff + drag_comp
        )

        a_cmd = clamp(a_pid, -self.max_acc_cmd, self.max_acc_cmd)

        # Add gravity compensation
        a_des = a_cmd + np.array([0.0, 0.0, self.g])

        # ------------------------------------------------------------
        # 4) Convert desired accel + yaw into attitude + thrust
        # ------------------------------------------------------------
        q_des, thrust_total = self.accel_to_attitude_and_thrust(a_des, self.desired_yaw)

        # ------------------------------------------------------------
        # 5) Attitude loop: quaternion error -> desired body rates
        # ------------------------------------------------------------
        w_ref = self.attitude_controller(q_des, q)
        w_ref = clamp(w_ref, -self.max_rate_cmd, self.max_rate_cmd)

        # ------------------------------------------------------------
        # 6) Rate loop: desired rates -> torques
        # ------------------------------------------------------------
        rate_error = w_ref - w

        tau_cmd = self.rate_pid.update(
            error=rate_error,
            dt=self.dt,
            measurement=w,
            feedforward=np.zeros(3)
        )

        tau_cmd = tau_cmd * self.J
        tau_cmd = clamp(tau_cmd, -self.max_torque_cmd, self.max_torque_cmd)

        # ------------------------------------------------------------
        # 7) Mixer: thrust + torques -> motor commands
        # ------------------------------------------------------------
        u = self.mix_to_motors(thrust_total, tau_cmd, quaternion_to_euler(q))

        # Smooth motor commands
        u = lowpass(self.prev_u, u, self.motor_smoothing_alpha)
        u = clamp(u, self.min_u, self.max_u)
        self.prev_u = u.copy()

        return u

    # ============================================================
    # Core blocks
    # ============================================================

    def compute_feedforward(self, local_trajectory):
        """
        Estimate feedforward velocity and acceleration from trajectory points.
        """
        if len(local_trajectory) < 3:
            return np.zeros(3), np.zeros(3)

        p0 = local_trajectory[0]
        p1 = local_trajectory[min(1, len(local_trajectory)-1)]
        p2 = local_trajectory[min(2, len(local_trajectory)-1)]

        v_ff = (p1 - p0) / self.dt
        v_ff = clamp(v_ff, -self.max_vel_cmd, self.max_vel_cmd)

        a_ff = (p2 - 2*p1 + p0) / (self.dt ** 2)
        a_ff = clamp(a_ff, -self.max_acc_cmd, self.max_acc_cmd)

        return v_ff, a_ff

    def accel_to_attitude_and_thrust(self, a_des, yaw_des):
        """
        Build desired body orientation such that body +Z aligns with desired acceleration.
        This matches your thrust model:
            body +Z thrust -> world acceleration
        """

        zb_des, a_norm = norm_safe(a_des)
        if a_norm < 1e-6:
            zb_des = np.array([0.0, 0.0, 1.0])

        # Desired heading in world frame
        xc = np.array([np.cos(yaw_des), np.sin(yaw_des), 0.0])

        yb_des, yn = norm_safe(np.cross(zb_des, xc))
        if yn < 1e-6:
            yb_des = np.array([0.0, 1.0, 0.0])

        xb_des, _ = norm_safe(np.cross(yb_des, zb_des))

        R_des = np.column_stack((xb_des, yb_des, zb_des))

        # Limit tilt
        z_world = np.array([0.0, 0.0, 1.0])
        tilt = np.arccos(clamp(np.dot(R_des[:, 2], z_world), -1.0, 1.0))
        if tilt > self.max_tilt:
            # Scale lateral accel down while preserving vertical
            lateral = a_des[:2]
            lat_norm = np.linalg.norm(lateral)
            max_lat = np.tan(self.max_tilt) * max(a_des[2], 1e-3)
            if lat_norm > max_lat and lat_norm > 1e-6:
                a_des[:2] = lateral * (max_lat / lat_norm)

            zb_des, _ = norm_safe(a_des)
            yb_des, yn = norm_safe(np.cross(zb_des, xc))
            if yn < 1e-6:
                yb_des = np.array([0.0, 1.0, 0.0])
            xb_des, _ = norm_safe(np.cross(yb_des, zb_des))
            R_des = np.column_stack((xb_des, yb_des, zb_des))

        q_des = rotmat_to_quat_wxyz(R_des)

        # Since body +Z produces thrust, total thrust should match projected desired accel magnitude
        thrust_total = self.mass * np.linalg.norm(a_des)
        thrust_total = clamp(thrust_total, 0.0, self.max_total_thrust)

        return q_des, thrust_total

    def attitude_controller(self, q_des, q):
        """
        Quaternion attitude control:
            q_err = q_des * q^{-1}
        Small-angle approximation:
            omega_cmd ≈ 2 * Kp * q_err_xyz
        """
        q_err = quat_error(q_des, q)
        rotvec_err = 2.0 * q_err[1:4]
        w_ref = self.att_kp * rotvec_err

        # Better yaw behavior: wrap yaw softly if near hover
        return w_ref

    def mix_to_motors(self, thrust_total, w_dot_cmd, w):
        """
        Dynamics-consistent motor mixer based directly on:

            w_dot_x = (-sum(f_i * y_i) + (Jy - Jz) * q * r) / Jx
            w_dot_y = ( sum(f_i * x_i) + (Jz - Jx) * r * p) / Jy
            w_dot_z = (-sum(f_i * c_i) + (Jx - Jy) * p * q) / Jz

        and total thrust:
            T = f1 + f2 + f3 + f4

        Parameters
        ----------
        thrust_total : float
            Desired total thrust in Newtons.
        w_dot_cmd : np.ndarray shape (3,)
            Desired body angular accelerations [p_dot, q_dot, r_dot] in rad/s^2.
        w : np.ndarray shape (3,)
            Current body rates [p, q, r] in rad/s.

        Returns
        -------
        u : np.ndarray shape (4,)
            Normalized motor commands such that:
                f_i = u_i * self.max_motor_thrust
        """

        p, q, r = w
        Jx, Jy, Jz = self.J

        # Build linear allocation matrix directly from dynamics
        # A @ f = b
        A = np.array([
            [1.0, 1.0, 1.0, 1.0],
            [-self.y_f[0], -self.y_f[1], -self.y_f[2], -self.y_f[3]],
            [ self.x_f[0],  self.x_f[1],  self.x_f[2],  self.x_f[3]],
            [-self.c_f[0], -self.c_f[1], -self.c_f[2], -self.c_f[3]],
        ], dtype=float)

        # Right-hand side from full rigid-body dynamics
        b = np.array([
            thrust_total,
            Jx * w_dot_cmd[0] - (Jy - Jz) * q * r,
            Jy * w_dot_cmd[1] - (Jz - Jx) * r * p,
            Jz * w_dot_cmd[2] - (Jx - Jy) * p * q
        ], dtype=float)

        # Solve for per-motor thrusts
        try:
            f = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            f = np.linalg.pinv(A) @ b

        # Saturate physical thrusts
        f = clamp(f, 0.0, self.max_motor_thrust)

        # Convert to normalized motor commands
        u = f / self.max_motor_thrust
        u = clamp(u, self.min_u, self.max_u)

        return u