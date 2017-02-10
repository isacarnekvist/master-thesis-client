from time import sleep
from arm_wrapper import Arm
import numpy as np

def sample(arm):
    arm.move_to(0.0, 0.2, 0.04)
    sleep(1.0)
    commanded = []
    real = []
    
    for i in range(8):
        x_before, y_before, _ = arm.get_position()
        dx, dy = 2 * (np.random.rand(2) * 0.04) - 0.04
        arm.move_to(x_before + dx, y_before + dy, 0.04)
        sleep(1.0)
        x_after, y_after, _ = arm.get_position()
        commanded.append((dx, dy))
        real.append((x_after - x_before, y_after - y_before))

    return commanded, real
