import pickle
import numpy as np


def first_quadrant(theta):
    theta -= np.pi
    while theta < 0.0:
        theta += np.pi / 2
    return theta


def lidar_to_arm_mapping(X_lidar, Y_arm):
    X = np.ones((len(X_lidar), 3))
    X[:, :2] = X_lidar
    Y = np.array(Y_arm)
    A, _, _, _ = np.linalg.lstsq(X, Y)
    return A[:2, :], A[-1, :]


def lidar_rotation_to_arm(theta_lidar, theta_arm):
    X = np.array([
        np.cos(theta_lidar),
        np.sin(theta_lidar),
    ]).T
    Y = np.array([
        np.cos(theta_arm),
        np.sin(theta_arm),
    ]).T
    A, _, _, _ = np.linalg.lstsq(X, Y)
    return A


if __name__ == '__main__':
    from time import sleep
    import numpy as np
    from arm_wrapper import Arm
    from pose_estimator import cube_pose, lw

    arm = Arm()
    sleep(5.0)
    print('Collecting positions, fasten cube to eef')
    arm.move_to(0.0, 0.20, 0.06)
    arm.set_pump(1.0)
    sleep(5.0)

    X_lidar = []
    X_arm = []
    theta_lidar = []
    theta_arm = []

    for x, y in np.random.rand(64, 2) * np.array([0.30, 0.15]) + np.array([-0.15, 0.13]):
        arm.move_to(x, y, 0.08, velocity=0.5)
        sleep(3.0)
        arm.move_to(x, y, 0.04)
        sleep(2.0)
        ax, ay, _ = arm.get_position()
        atheta = first_quadrant(np.arctan2(ay, ax))
        res = cube_pose()
        if res is None:
            continue
        cx, cy, ctheta = res
        ctheta = first_quadrant(ctheta)
        print('arm:', ax, ay, np.cos(atheta), np.sin(atheta))
        print('lidar:', cx, cy, np.cos(ctheta), np.sin(atheta))
        print()
        X_arm.append((ax, ay))
        theta_arm.append(atheta)
        X_lidar.append((cx, cy))
        theta_lidar.append(ctheta)
    
    A, b = lidar_to_arm_mapping(X_lidar, X_arm)
    Ar = lidar_rotation_to_arm(theta_lidar, theta_arm)
    with open('lidar_transform.pkl', 'wb') as f:
        pickle.dump((A, b, Ar), f)
    print('A', A)
    print('b', b)
    print('Ar', Ar)

    arm.set_pump(0.0)
    arm.stop()
    arm.disconnect()
    lw.stop()
    exit(0)
