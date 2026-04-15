# Quadrotor Formation using Model Predictive Control

## ROS 2 PX4 MPC Node
This repository now includes a ROS 2 package `quadrotor_acados` that:
- subscribes to `nav_msgs/msg/Path` (reference trajectory),
- subscribes to `px4_msgs/msg/VehicleOdometry` (current state),
- publishes `px4_msgs/msg/ActuatorMotors` (motor command).

The node uses `quadrotor_acados/config/x500.yaml` by default.

Build and run:
```bash
colcon build --packages-select quadrotor_acados
source install/setup.bash
ros2 launch quadrotor_acados px4_mpc.launch.py
```

### Python dependencies
Install pip dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

ROS 2 Python modules used by this package are provided by ROS packages (not pip), including:
- `rclpy`
- `ament_index_python`
- `geometry_msgs`
- `nav_msgs`
- `px4_msgs`
- `launch`
- `launch_ros`

Install ROS dependencies with:
```bash
rosdep install --from-paths . --ignore-src -r -y
```

Important runtime parameters:
- Topic names and node runtime parameters are defined in `quadrotor_acados/config/px4_mpc_node.yaml`.
- `path_topic` (default from config: `/reference_path`)
- `odometry_topic` (default from config: `/fmu/out/vehicle_odometry`)
- `actuator_topic` (default from config: `/fmu/in/actuator_motors`)
- `quadrotor_params_file` (default: installed `x500.yaml`)

If you run the node directly, load the same config explicitly:
```bash
ros2 run quadrotor_acados px4_mpc_node --ros-args --params-file $(ros2 pkg prefix quadrotor_acados)/share/quadrotor_acados/config/px4_mpc_node.yaml
```

## Sample Trajectory Publisher
The package includes a sample ROS 2 node that publishes a square reference trajectory as `nav_msgs/msg/Path`.

Run:
```bash
ros2 run quadrotor_acados square_path_publisher
```

## Install Acados
To build Acados from source, see instructions [here](https://docs.acados.org/python_interface/index.html) or as follows:

Clone acados and its submodules by running:
```
$ git clone https://github.com/acados/acados.git
$ cd acados
$ git submodule update --recursive --init
```

Install acados as follows:

```
$ mkdir -p build
$ cd build
$ cmake -DACADOS_WITH_QPOASES=ON ..
$ make install -j4
```

Install acados_template Python package:
```
$ cd acados
$ pip install -e interfaces/acados_template
```
***Note:*** The ```<acados_root>``` is the full path from ```/home/```.

Add two paths below to ```~/.bashrc``` in order to add the compiled shared libraries ```libacados.so```, ```libblasfeo.so```, ```libhpipm.so``` to ```LD_LIBRARY_PATH``` (default path is ```<acados_root/lib>```):

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"<acados_root>/lib"
export ACADOS_SOURCE_DIR="<acados_root>"
```

## Quadrotor Dynamics 
The full explanation of the quadrotor dynamics is presented in [here](https://rpg.ifi.uzh.ch/docs/ScienceRobotics21_Foehn.pdf).

The quadrotor state space is described between the inertial frame $I$ and body frame $B$, as $`\xi = \left[\begin{array}{cccc}p_{IB} & q_{IB} & v_{IB} & \omega_{B}\end{array}\right]^T`$ corresponding to position $`p_{IB} ∈ \mathbb{R}^3`$, unit quaternion rotation on the rotation group $`q_{IB} \in \mathbb{SO}(3)`$ given $`\left\Vert q_{IB}\right\Vert = 1`$, velocity $`v_{IB} \in \mathbb{R}^3`$, and bodyrate $`\omega_B \in \mathbb{R}^3`$. The input modality is on the level of collective thrust $`T_B = \left[\begin{array}{ccc}0 & 0 & T_{Bz} \end{array}\right]^T`$ and body torque $`\tau_B`$ . From here on we drop the frame indices since they are consistent throughout the description. The dynamic equations are follows:
```math
\dot{p}=v \\
\dot{q}= \dfrac{1}{2}\Lambda(q)\left[\begin{array}{l}0\\\omega\end{array}\right] \\
\dot{v}=g+\dfrac{1}{m}R(q)T\\
\dot{\omega}=J^{-1}(\tau-\omega\times J\omega)
```
where $`\Lambda`$ represents a quaternion multiplication, $`R(q)`$ the quaternion rotation, $m$ the quadrotor’s mass, and $`J`$ its inertia.

The input space given by $`T`$ and $`\tau`$ is further decomposed into the single rotor thrusts $`u =\left[T_1, T_2, T_3, T_4\right]^T`$, where $`T_i`$ is the thrust at rotor $`i \in \{1, 2, 3, 4\}`$
```math
T=\left[\begin{array}{c}0\\0\\\sum{T_i}\end{array}\right]
```
For $`\times`$ configuration quadrotor model
```math
\tau=\left[\begin{array}{c}l/\sqrt{2}(T_1-T_2-T_3+T_4)\\
                           l/\sqrt{2}(-T_1-T_2+T_3+T_4)\\
                           c_\tau(-T_1+T_2-T_3+T_4)\end{array}\right]
```
For $`+`$ configuration quadrotor model
```math
\tau=\left[\begin{array}{c}l(T_1-T_3)\\
                           l(T_2-T_4)\\
                           c_\tau(-T_1+T_2-T_3+T_4)\end{array}\right]
```
with the quadrotor’s arm length $l$ and the rotor’s torque constant $`c_\tau`$. The quadrotor’s actuators limit the applicable thrust for each rotor, effectively constraining $`T_i`$ as:
```math
0\leq T_{min} \leq T_i \leq T_{max}
```


## Results
### Control performance
Moving to goal           |   Trajectory tracking 
:-----------------------:|:-------------------------:
![](results/result.png)  |  ![](results/tracking.png)

### CPU time
```
ave estimation time is 0.00075
max estimation time is 0.00104
min estimation time is 0.00070
```
![](results/time.png)
