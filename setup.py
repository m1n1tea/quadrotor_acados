from glob import glob

from setuptools import find_packages, setup

package_name = "quadrotor_acados"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            "share/" + package_name + "/config",
            ["config/x500.yaml", "config/px4_mpc_node.yaml"],
        ),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools", "numpy", "PyYAML", "casadi"],
    extras_require={
        # acados Python bindings (acados_template) are typically installed
        # from an acados source checkout; keep this optional.
        "acados": ["acados_template"],
    },
    zip_safe=True,
    maintainer="quadrotor_acados",
    maintainer_email="maintainer@example.com",
    description="ROS 2 PX4 MPC bridge for quadrotor control",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "px4_mpc_node = quadrotor_acados.ros2_px4_mpc_node:main",
            "square_path_publisher = quadrotor_acados.square_path_publisher:main",
        ],
    },
)
