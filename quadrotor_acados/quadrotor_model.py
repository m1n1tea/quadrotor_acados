from dataclasses import dataclass, field
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
    thrust_max: float
    torque_coeff: float
    configuration: str = "x"

    length: float = field(init=False)
    J: np.ndarray = field(init=False)
    max_thrust: float = field(init=False)
    min_thrust: float = field(init=False)
    max_input_value: float = field(init=False, default=1.0)
    min_input_value: float = field(init=False, default=0.0)
    x_f: np.ndarray = field(init=False)
    y_f: np.ndarray = field(init=False)
    z_l_tau: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.length = self.arm_length
        self.J = np.array([self.inertia_xx, self.inertia_yy, self.inertia_zz], dtype=float)

        self.max_thrust = self.thrust_max
        self.min_thrust = self.thrust_min

        if self.configuration == "+":
            self.x_f = np.array([self.length, 0.0, -self.length, 0.0], dtype=float)
            self.y_f = np.array([0.0, self.length, 0.0, -self.length], dtype=float)
        else:
            h = np.cos(np.pi / 4.0) * self.length
            self.x_f = np.array([h, -h, -h, h], dtype=float)
            self.y_f = np.array([h, -h, h, -h], dtype=float)

        c = self.torque_coeff
        self.z_l_tau = np.array([-c, c, -c, c], dtype=float)


def load_params(params_file: str) -> QuadrotorParams:
    params_path = Path(params_file)
    with params_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if "/**" in data:
        data = data["/**"]

    ros_params = data.get("ros__parameters", data)
    uav = ros_params["uav"]["parameters"]
    inertia = uav["inertia"]

    mass = float(uav["uav_mass"])
    arm_length = float(uav["arm_length"])
    thrust_constant = float(uav["thrust_constant"])
    min_rotor_speed = float(uav.get("min_rotor_speed", 0.0))
    max_rotor_speed = float(uav["max_rotor_speed"])
    thrust_max = thrust_constant * (max_rotor_speed**2)
    thrust_min = thrust_constant * (min_rotor_speed**2)

    return QuadrotorParams(
        mass=mass,
        arm_length=arm_length,
        inertia_xx=float(inertia["xx"]),
        inertia_yy=float(inertia["yy"]),
        inertia_zz=float(inertia["zz"]),
        thrust_min=thrust_min,
        thrust_max=thrust_max,
        torque_coeff=float(uav["moment_constant"]),
    )
