from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    package_share = Path(get_package_share_directory("quadrotor_acados"))
    params_file = package_share / "config" / "px4_mpc_node.yaml"

    return LaunchDescription(
        [
            Node(
                package="quadrotor_acados",
                executable="px4_mpc_node",
                name="px4_mpc_node",
                output="screen",
                parameters=[str(params_file)],
            )
        ]
    )
