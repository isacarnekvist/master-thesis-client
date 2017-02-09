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

    for x in np.linspace(-0.09, 0.09, 4):
        for y in np.linspace(0.15, 0.25, 4):
            arm.move_to(x, y, 0.06, velocity=0.5)
            sleep(3.0)
            arm.move_to(x, y, 0.04)
            sleep(2.0)
            xp, yp, zp = arm.get_position()
            print(xp, yp, cube_pose())

    arm.set_pump(0.0)
    arm.stop()
    arm.disconnect()
    lw.stop()
    exit(0)
