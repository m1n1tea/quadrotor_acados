import numpy as np
import math
import timeit
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D 

from quadrotor import Quadrotor3D
from controller import Controller

from utils import quaternion_to_euler, transform_trajectory

def createTrajectory(points_count):
    xref = []
    yref = []
    zref = []
    radius = 5.0
    height = -100.0  # Modified to make z negative
    offset_height = -1.0  # Offset height
    num_turns = 2
    for i in range(points_count):
        t = i / points_count
        x = radius * math.cos(2 * math.pi * num_turns * t)
        y = radius * math.sin(2 * math.pi * num_turns * t)
        z = height * t + offset_height  # Add offset height
        xref.append(x)
        yref.append(y)
        zref.append(z)
    return np.array(xref), np.array(yref), np.array(zref)

def trackTrajectory():
    dt = 0.1      # Time step
    sim_time = 20 # Simulation time
    N = 100       # Horizontal length
    
    quad = Quadrotor3D()    # Quadrotor model
    controller = Controller(quad, t_horizon=2*N*dt, n_nodes=N)  # Initialize MPC controller
    xref, yref, zref = createTrajectory(1000)
    trajectory = np.array([xref, yref, zref]).T
    controller.update_trajectory(trajectory, preferred_speed=8)
    path = []
    q_path = []
    u_path = []

    # Main loop
    time_record = []
    for i in range(int(sim_time/dt)):
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

    # Visualization
    path = np.array(path)
    # print(path)
    plt.figure()
    plt.title('UAV Position in ENU frame')
    ax = plt.axes(projection='3d')
    ax.plot(yref, xref, -zref, c=[1,0,0], label='goal')
    ax.plot(path[:,1], path[:,0], -path[:,2])  # Swap x and y, negate z
    ax.axis('auto')
    ax.set_xlabel('y [m]')  # Swap x and y labels
    ax.set_ylabel('x [m]')
    ax.set_zlabel('-z [m]')  # Negate z label
    ax.legend()

    # Plot UAV attitude axes
    interval = 50
    for i in range(0, len(q_path), interval):
        euler_angles = quaternion_to_euler(q_path[i])
        origin = path[i]
        R = np.array([[math.cos(euler_angles[2])*math.cos(euler_angles[1]), math.cos(euler_angles[2])*math.sin(euler_angles[1])*math.sin(euler_angles[0])-math.sin(euler_angles[2])*math.cos(euler_angles[0]), math.cos(euler_angles[2])*math.sin(euler_angles[1])*math.cos(euler_angles[0])+math.sin(euler_angles[2])*math.sin(euler_angles[0])],
                  [math.sin(euler_angles[2])*math.cos(euler_angles[1]), math.sin(euler_angles[2])*math.sin(euler_angles[1])*math.sin(euler_angles[0])+math.cos(euler_angles[2])*math.cos(euler_angles[0]), math.sin(euler_angles[2])*math.sin(euler_angles[1])*math.cos(euler_angles[0])-math.cos(euler_angles[2])*math.sin(euler_angles[0])],
                  [-math.sin(euler_angles[1]), math.cos(euler_angles[1])*math.sin(euler_angles[0]), math.cos(euler_angles[1])*math.cos(euler_angles[0])]])
        x_axis = R @ np.array([1, 0, 0])
        y_axis = R @ np.array([0, 1, 0])
        z_axis = R @ np.array([0, 0, 1])
        ax.quiver(origin[1], origin[0], -origin[2], y_axis[1], y_axis[0], -y_axis[2], color='r', length=0.05)  # Swap x and y, negate z
        ax.quiver(origin[1], origin[0], -origin[2], x_axis[1], x_axis[0], -x_axis[2], color='g', length=0.05)  # Swap x and y, negate z
        ax.quiver(origin[1], origin[0], -origin[2], -z_axis[1], -z_axis[0], z_axis[2], color='b', length=0.05)  # Swap x and y, negate z

    plt.figure()
    plt.plot(time_record)
    plt.legend()
    plt.ylabel('CPU Time [s]')
    # plt.yscale("log")

    # Visualize inputs
    u_path = np.array(u_path)
    time = np.arange(0, len(u_path)*dt, dt)
    plt.figure()
    plt.suptitle("Rotor thrust - normalized")
    plt.subplot(2, 2, 1)
    plt.plot(time, u_path[:, 0])
    plt.ylabel('u1')
    plt.xlabel('Time [s]')
    plt.subplot(2, 2, 2)
    plt.plot(time, u_path[:, 1])
    plt.ylabel('u2')
    plt.xlabel('Time [s]')
    plt.subplot(2, 2, 3)
    plt.plot(time, u_path[:, 2])
    plt.ylabel('u3')
    plt.xlabel('Time [s]')
    plt.subplot(2, 2, 4)
    plt.plot(time, u_path[:, 3])
    plt.ylabel('u4')
    plt.xlabel('Time [s]')
    plt.tight_layout()
    
    # Visualize quaternion
    q_path = np.array(q_path)
    plt.figure()
    plt.suptitle("UAV attitude")
    plt.subplot(2, 2, 1)
    plt.plot(time, q_path[:, 0])
    plt.ylabel('qw')
    plt.xlabel('Time [s]')
    plt.subplot(2, 2, 2)
    plt.plot(time, q_path[:, 1])
    plt.ylabel('qx')
    plt.xlabel('Time [s]')
    plt.subplot(2, 2, 3)
    plt.plot(time, q_path[:, 2])
    plt.ylabel('qy')
    plt.xlabel('Time [s]')
    plt.subplot(2, 2, 4)
    plt.plot(time, q_path[:, 3])
    plt.ylabel('qz')
    plt.xlabel('Time [s]')
    plt.tight_layout()
    
    plt.show()

if __name__ == "__main__":
    # move2Goal()
    trackTrajectory()