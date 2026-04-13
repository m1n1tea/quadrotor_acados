from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass
class QuadrotorParams:
    mass: float
    arm_length: float
    inertia_xx: float
    inertia_yy: float
    inertia_zz: float
    thrust_min: float
    thrust_to_weight_max: float
    torque_coeff: float


class Quadrotor3D:
    def __init__(self, params: QuadrotorParams, configuration: str = "x"):
        self.mass = params.mass
        self.length = params.arm_length
        self.J = np.array([params.inertia_xx, params.inertia_yy, params.inertia_zz])

        max_total_thrust = self.mass * 9.81 * params.thrust_to_weight_max
        self.max_thrust = -(max_total_thrust / 4.0)
        self.min_thrust = params.thrust_min

        self.max_input_value = 1.0
        self.min_input_value = 0.0

        if configuration == "+":
            self.x_f = np.array([self.length, 0.0, -self.length, 0.0])
            self.y_f = np.array([0.0, self.length, 0.0, -self.length])
        else:
            h = np.cos(np.pi / 4.0) * self.length
            self.x_f = np.array([h, -h, -h, h])
            self.y_f = np.array([-h, -h, h, h])

        c = params.torque_coeff
        self.z_l_tau = np.array([-c, c, -c, c])



def load_params(params_file: str) -> QuadrotorParams:
    params_path = Path(params_file)
    with params_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    quad = data["quadrotor"]
    inertia = quad["inertia"]

    return QuadrotorParams(
        mass=float(quad["mass"]),
        arm_length=float(quad["arm_length"]),
        inertia_xx=float(inertia["xx"]),
        inertia_yy=float(inertia["yy"]),
        inertia_zz=float(inertia["zz"]),
        thrust_min=float(quad["thrust_min"]),
        thrust_to_weight_max=float(quad["thrust_to_weight_max"]),
        torque_coeff=float(quad["torque_coeff"]),
    )
