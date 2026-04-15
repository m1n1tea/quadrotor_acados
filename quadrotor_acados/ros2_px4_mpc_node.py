from pathlib import Path
from threading import Lock

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Path as PathMsg
from px4_msgs.msg import (
    ActuatorMotors,
    OffboardControlMode,
    VehicleCommand,
    VehicleOdometry,
    VehicleStatus,
)
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from .mpc_controller import Controller
from .quadrotor_model import load_params


class Px4MpcNode(Node):
    def __init__(self):
        super().__init__("px4_mpc_node")

        package_share = Path(get_package_share_directory("quadrotor_acados"))
        default_params = package_share / "config" / "x500.yaml"

        self.declare_parameter("path_topic", "/reference_path")
        self.declare_parameter("odometry_topic", "/fmu/out/vehicle_odometry")
        self.declare_parameter("vehicle_status_topic", "/fmu/out/vehicle_status")
        self.declare_parameter("actuator_topic", "/fmu/in/actuator_motors")
        self.declare_parameter(
            "offboard_control_mode_topic", "/fmu/in/offboard_control_mode"
        )
        self.declare_parameter("vehicle_command_topic", "/fmu/in/vehicle_command")
        self.declare_parameter("control_rate_hz", 50.0)
        self.declare_parameter("offboard_control_rate_hz", 3.0)
        self.declare_parameter("preferred_speed", 1.0)
        self.declare_parameter("horizon_sec", 2.0)
        self.declare_parameter("horizon_nodes", 20)
        self.declare_parameter("quadrotor_params_file", str(default_params))

        self.path_topic = self.get_parameter("path_topic").value
        self.odom_topic = self.get_parameter("odometry_topic").value
        self.vehicle_status_topic = self.get_parameter("vehicle_status_topic").value
        self.actuator_topic = self.get_parameter("actuator_topic").value
        self.offboard_control_mode_topic = self.get_parameter(
            "offboard_control_mode_topic"
        ).value
        self.vehicle_command_topic = self.get_parameter("vehicle_command_topic").value
        self.control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.offboard_control_rate_hz = float(
            self.get_parameter("offboard_control_rate_hz").value
        )
        self.preferred_speed = float(self.get_parameter("preferred_speed").value)
        horizon_sec = float(self.get_parameter("horizon_sec").value)
        horizon_nodes = int(self.get_parameter("horizon_nodes").value)
        params_file = str(self.get_parameter("quadrotor_params_file").value)

        self.lock = Lock()
        self.current_state = None
        self.is_armed = False
        self.offboard_setpoint_counter = 0
        self.arm_sequence_sent = False
        self.last_state_log_time_sec = -1.0

        try:
            quad = load_params(params_file)
            self.controller = Controller(
                quad=quad,
                t_horizon=horizon_sec,
                n_nodes=horizon_nodes,
                logger = self.get_logger()
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

        self.path_sub = self.create_subscription(
            PathMsg, self.path_topic, self.path_callback, 10
        )
        self.odom_sub = self.create_subscription(
            VehicleOdometry, self.odom_topic, self.odom_callback, qos_sensor
        )
        self.vehicle_status_sub = self.create_subscription(
            VehicleStatus,
            self.vehicle_status_topic,
            self.vehicle_status_callback,
            qos_sensor,
        )
        self.motor_pub = self.create_publisher(
            ActuatorMotors, self.actuator_topic, qos_sensor
        )
        self.offboard_control_mode_pub = self.create_publisher(
            OffboardControlMode, self.offboard_control_mode_topic, qos_sensor
        )
        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand, self.vehicle_command_topic, qos_sensor
        )

        self.control_timer = self.create_timer(
            1.0 / self.control_rate_hz, self.control_loop
        )
        self.offboard_mode_timer = self.create_timer(
            1.0 / self.offboard_control_rate_hz, self.publish_offboard_control_mode
        )

        self.get_logger().info(
            "px4_mpc_node started. "
            f"path={self.path_topic}, odom={self.odom_topic}, vehicle_status={self.vehicle_status_topic}, "
            f"actuator={self.actuator_topic}, offboard_control_mode={self.offboard_control_mode_topic}, "
            f"vehicle_command={self.vehicle_command_topic}"
        )

    def path_callback(self, msg: PathMsg) -> None:
        if not msg.poses:
            return

        trajectory = np.array(
            [
                [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z]
                for pose in msg.poses
            ],
            dtype=float,
        )

        with self.lock:
            self.controller.update_trajectory(
                trajectory, preferred_speed=self.preferred_speed
            )

    def odom_callback(self, msg: VehicleOdometry) -> None:
        position = np.array(
            [msg.position[0], msg.position[1], msg.position[2]], dtype=float
        )
        quat = np.array([msg.q[0], msg.q[1], msg.q[2], msg.q[3]], dtype=float)
        velocity = np.array(
            [msg.velocity[0], msg.velocity[1], msg.velocity[2]], dtype=float
        )
        angular_velocity = np.array(
            [msg.angular_velocity[0], msg.angular_velocity[1], msg.angular_velocity[2]],
            dtype=float,
        )

        state = np.concatenate([position, quat, velocity, angular_velocity])

        with self.lock:
            self.current_state = state

    def vehicle_status_callback(self, msg: VehicleStatus) -> None:
        if not hasattr(msg, "arming_state"):
            return

        armed_state = getattr(VehicleStatus, "ARMING_STATE_ARMED", 2)
        is_armed_now = msg.arming_state == armed_state
        with self.lock:
            if is_armed_now == self.is_armed:
                return
            self.is_armed = is_armed_now

        self.get_logger().info(f"Vehicle armed state changed: armed={self.is_armed}")

    def publish_offboard_control_mode(self) -> None:
        msg = OffboardControlMode()
        if hasattr(msg, "timestamp"):
            msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

        # Keep all modes disabled except direct actuator control.
        for field in [
            "position",
            "velocity",
            "acceleration",
            "attitude",
            "body_rate",
            "thrust_and_torque",
        ]:
            if hasattr(msg, field):
                setattr(msg, field, False)
        if hasattr(msg, "direct_actuator"):
            msg.direct_actuator = True
        elif hasattr(msg, "actuator"):
            msg.actuator = True

        self.offboard_control_mode_pub.publish(msg)

        if self.arm_sequence_sent:
            return

        if self.offboard_setpoint_counter == 10:
            self.set_offboard_mode()
            self.arm()
            self.arm_sequence_sent = True
            self.get_logger().info("Offboard mode set, vehicle armed")

        if self.offboard_setpoint_counter < 11:
            self.offboard_setpoint_counter += 1

    def publish_vehicle_command(
        self, command: int, param1: float = 0.0, param2: float = 0.0
    ) -> None:
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        if hasattr(msg, "timestamp"):
            msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_pub.publish(msg)

    def arm(self) -> None:
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0
        )

    def set_offboard_mode(self) -> None:
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0
        )

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
