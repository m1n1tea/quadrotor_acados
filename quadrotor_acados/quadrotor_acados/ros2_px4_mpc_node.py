from pathlib import Path
from threading import Lock

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Path as PathMsg
from px4_msgs.msg import ActuatorMotors, VehicleOdometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from .mpc_controller import Controller
from .quadrotor_model import Quadrotor3D, load_params


class Px4MpcNode(Node):
    def __init__(self):
        super().__init__("px4_mpc_node")

        package_share = Path(get_package_share_directory("quadrotor_acados"))
        default_params = package_share / "config" / "uav_params.yaml"

        self.declare_parameter("path_topic", "/reference_path")
        self.declare_parameter("odometry_topic", "/fmu/out/vehicle_odometry")
        self.declare_parameter("actuator_topic", "/fmu/in/actuator_motors")
        self.declare_parameter("control_rate_hz", 50.0)
        self.declare_parameter("preferred_speed", 1.0)
        self.declare_parameter("horizon_sec", 2.0)
        self.declare_parameter("horizon_nodes", 20)
        self.declare_parameter("acados_build_dir", "/tmp/quadrotor_acados")
        self.declare_parameter("params_file", str(default_params))

        self.path_topic = self.get_parameter("path_topic").value
        self.odom_topic = self.get_parameter("odometry_topic").value
        self.actuator_topic = self.get_parameter("actuator_topic").value
        self.control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.preferred_speed = float(self.get_parameter("preferred_speed").value)
        horizon_sec = float(self.get_parameter("horizon_sec").value)
        horizon_nodes = int(self.get_parameter("horizon_nodes").value)
        acados_build_dir = str(self.get_parameter("acados_build_dir").value)
        params_file = str(self.get_parameter("params_file").value)

        self.lock = Lock()
        self.current_state = None

        try:
            quad = Quadrotor3D(load_params(params_file))
            self.controller = Controller(
                quad=quad,
                t_horizon=horizon_sec,
                n_nodes=horizon_nodes,
                acados_build_dir=acados_build_dir,
            )
        except Exception as exc:
            self.get_logger().error(f"Failed to initialize controller: {exc}")
            raise

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.path_sub = self.create_subscription(PathMsg, self.path_topic, self.path_callback, 10)
        self.odom_sub = self.create_subscription(VehicleOdometry, self.odom_topic, self.odom_callback, qos_sensor)
        self.motor_pub = self.create_publisher(ActuatorMotors, self.actuator_topic, qos_sensor)

        self.control_timer = self.create_timer(1.0 / self.control_rate_hz, self.control_loop)

        self.get_logger().info(
            f"px4_mpc_node started. path={self.path_topic}, odom={self.odom_topic}, actuator={self.actuator_topic}"
        )

    def path_callback(self, msg: PathMsg) -> None:
        if not msg.poses:
            return

        trajectory = np.array(
            [[pose.pose.position.x, pose.pose.position.y, pose.pose.position.z] for pose in msg.poses],
            dtype=float,
        )

        with self.lock:
            self.controller.update_trajectory(trajectory, preferred_speed=self.preferred_speed)

    def odom_callback(self, msg: VehicleOdometry) -> None:
        position = np.array([msg.position[0], msg.position[1], msg.position[2]], dtype=float)
        quat = np.array([msg.q[0], msg.q[1], msg.q[2], msg.q[3]], dtype=float)
        velocity = np.array([msg.velocity[0], msg.velocity[1], msg.velocity[2]], dtype=float)
        angular_velocity = np.array([msg.angular_velocity[0], msg.angular_velocity[1], msg.angular_velocity[2]], dtype=float)

        state = np.concatenate([position, quat, velocity, angular_velocity])

        with self.lock:
            self.current_state = state

    def control_loop(self) -> None:
        with self.lock:
            if self.current_state is None or self.controller.time_traj is None:
                return

            cmd = self.controller.run_optimization(initial_state=self.current_state)

        cmd = np.clip(np.array(cmd[:4], dtype=float), 0.0, 1.0)

        msg = ActuatorMotors()
        timestamp_us = int(self.get_clock().now().nanoseconds / 1000)
        if hasattr(msg, "timestamp"):
            msg.timestamp = timestamp_us
        if hasattr(msg, "timestamp_sample"):
            msg.timestamp_sample = timestamp_us
        if hasattr(msg, "reversible_flags"):
            msg.reversible_flags = 0

        control = [float("nan")] * 12
        control[0:4] = [float(cmd[0]), float(cmd[1]), float(cmd[2]), float(cmd[3])]
        msg.control = control

        self.motor_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = Px4MpcNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
