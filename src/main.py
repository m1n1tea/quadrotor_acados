import math
import timeit

import matplotlib
import numpy as np

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from controller import Controller
from pid_controller import PIDController
from quadrotor import Quadrotor3D
from utils import quaternion_to_euler, transform_trajectory


def createTrajectory(points_count):
    xref = []
    yref = []
    zref = []
    radius = 2.0
    height = -3.0
    offset_height = 0  # Offset height
    num_turns = 0
    x_offset = 1
    y_offset = 0.5
    for i in range(points_count):
        t = i / (points_count - 1)
        x = radius * math.cos(2 * math.pi * num_turns * t) + x_offset * t - radius
        y = radius * math.sin(2 * math.pi * num_turns * t) + y_offset * t
        z = height * t + offset_height  # Add offset height
        xref.append(x)
        yref.append(y)
        zref.append(z)
    return np.array([np.array(xref), np.array(yref), np.array(zref)]).T


def trackTrajectory(
    trajectory,
    preferred_speed=0.5,
    noise=False,
    controller_type="Acados",
):
    sim_time = 7  # Simulation time
    N = 20  # Horizontal length

    quad = Quadrotor3D(motor_noise=noise)  # Quadrotor model
    controller = None
    if controller_type == "Acados":
        dt = 0.1  # Time step
        controller = Controller(
            quad, t_horizon=2 * N * dt, n_nodes=N
        )  # Initialize MPC controller
    elif controller_type == "PID":
        dt = 0.001  # Time step
        controller = PIDController(quad, dt=dt)  # Initialize PID controller
    else:
        print("Invalid controller type")
        return
    xref = trajectory[:, 0]
    yref = trajectory[:, 1]
    zref = trajectory[:, 2]
    # xref, yref, zref = createTrajectory(10)
    trajectory = np.array([xref, yref, zref]).T
    controller.update_trajectory(trajectory, preferred_speed=preferred_speed)
    path = []
    q_path = []
    u_path = []

    # xref_cmp, yref_cmp, zref_cmp = createTrajectory(int(sim_time/dt))

    # Main loop
    time_record = []
    for i in range(int(sim_time / dt)):
        current = np.concatenate([quad.pos, quad.angle, quad.vel, quad.a_rate])
        start = timeit.default_timer()
        thrust = controller.run_optimization(initial_state=current)[:4]
        time_record.append(timeit.default_timer() - start)
        quad.update(thrust, dt)
        path.append(quad.pos)
        q_path.append(quad.angle)
        u_path.append(thrust)

    # CPU time
    print("average estimation time is {:.5f}".format(np.array(time_record).mean()))
    print("max estimation time is {:.5f}".format(np.array(time_record).max()))
    print("min estimation time is {:.5f}".format(np.array(time_record).min()))

    # reference_trajectory = np.array([xref_cmp, yref_cmp, zref_cmp]).T

    # # Приводим длины массивов к одинаковому размеру (на случай расхождений)
    # min_len = min(len(path), len(reference_trajectory))
    # path_trimmed = np.array(path)[:min_len]
    # ref_trimmed = reference_trajectory[:min_len]

    # # Вычисляем Евклидово расстояние для каждого шага времени
    # deviations = np.sqrt(np.sum((path_trimmed - ref_trimmed)**2, axis=1))

    # # Максимальное и среднее отклонение
    # max_deviation = np.max(deviations)
    # mean_deviation = np.mean(deviations)

    # print(f"Maximum deviation from path: {max_deviation:.4f} m")
    # print(f"Average deviation from path: {mean_deviation:.4f} m")

    # Visualization
    path = np.array(path)
    # print(path)

    plt.figure()
    plt.plot(time_record)
    plt.legend()
    plt.ylabel("CPU Time [s]")
    # plt.yscale("log")

    # Visualize inputs
    u_path = np.array(u_path)
    time = np.arange(0, len(u_path) * dt, dt)
    plt.figure()
    plt.suptitle("Rotor thrust - normalized")
    plt.subplot(2, 2, 1)
    plt.plot(time, u_path[:, 0])
    plt.ylabel("u1")
    plt.xlabel("Time [s]")
    plt.subplot(2, 2, 2)
    plt.plot(time, u_path[:, 1])
    plt.ylabel("u2")
    plt.xlabel("Time [s]")
    plt.subplot(2, 2, 3)
    plt.plot(time, u_path[:, 2])
    plt.ylabel("u3")
    plt.xlabel("Time [s]")
    plt.subplot(2, 2, 4)
    plt.plot(time, u_path[:, 3])
    plt.ylabel("u4")
    plt.xlabel("Time [s]")
    plt.tight_layout()

    # Visualize quaternion
    q_path = np.array(q_path)
    plt.figure()
    plt.suptitle("UAV attitude")
    plt.subplot(2, 2, 1)
    plt.plot(time, q_path[:, 0])
    plt.ylabel("qw")
    plt.xlabel("Time [s]")
    plt.subplot(2, 2, 2)
    plt.plot(time, q_path[:, 1])
    plt.ylabel("qx")
    plt.xlabel("Time [s]")
    plt.subplot(2, 2, 3)
    plt.plot(time, q_path[:, 2])
    plt.ylabel("qy")
    plt.xlabel("Time [s]")
    plt.subplot(2, 2, 4)
    plt.plot(time, q_path[:, 3])
    plt.ylabel("qz")
    plt.xlabel("Time [s]")
    plt.tight_layout()

    # q_path = np.array(q_path)
    # plt.figure()
    # plt.suptitle("XYZ trajectory")
    # plt.subplot(2, 2, 1)
    # plt.plot(time, path[:,0])
    # plt.ylabel('X')
    # plt.xlabel('Time [s]')
    # plt.plot(time, xref_cmp)
    # plt.subplot(2, 2, 2)
    # plt.plot(time, path[:,1])
    # plt.plot(time, yref_cmp)
    # plt.ylabel('Y')
    # plt.xlabel('Time [s]')
    # plt.subplot(2, 2, 3)
    # plt.plot(time, path[:,2])
    # plt.plot(time, zref_cmp)
    # plt.ylabel('Z')
    # plt.xlabel('Time [s]')
    # plt.tight_layout()

    return path


if __name__ == "__main__":
    # move2Goal()
    gt_traj = createTrajectory(100)
    pid_traj = trackTrajectory(createTrajectory(2), controller_type="PID")
    acados_traj = trackTrajectory(
        createTrajectory(10), controller_type="Acados", noise=True, preferred_speed=0.9
    )
    print(gt_traj)

    plt.figure()
    plt.title("Визуализация траекторий дрона")
    ax = plt.axes(projection="3d")
    ax.plot(
        gt_traj[:, 0],
        gt_traj[:, 1],
        gt_traj[:, 2],
        c=[0, 0.7, 0],
        label="Референсная траектория",
    )
    ax.plot(
        acados_traj[:, 0],
        acados_traj[:, 1],
        acados_traj[:, 2],
        c=[0, 0, 1],
        label="Решение Acados контроллера",
    )
    ax.plot(
        pid_traj[:, 0],
        pid_traj[:, 1],
        pid_traj[:, 2],
        c=[1, 0, 0],
        label="Решение PID контроллера",
    )
    ax.axis("auto")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend()
    plt.show()
