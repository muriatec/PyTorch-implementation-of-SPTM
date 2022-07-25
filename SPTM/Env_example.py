import os
from random import choice
import yaml
import numpy as np
import cv2 as cv
import igibson
from igibson.envs.igibson_env import iGibsonEnv

config_data = yaml.load(open("/home/charles-chen/iGibson/igibson/configs/turtlebot_nav.yaml", "r"), Loader=yaml.FullLoader)
'''
scene = ['Beechwood_0_int', 'Benevolence_0_int', 'Benevolence_1_int', 'Benevolence_2_int', 'Ihlen_0_int',
         'Merom_0_int', 'Merom_1_int', 'Pomaria_1_int', 'Pomaria_2_int ', 'Rs_int', 'Wainscott_0_int',
         'Wainscott_1_int']
'''
scene = ['Rs_int']

env = iGibsonEnv(config_file=config_data, scene_id=choice(scene), mode="gui_non_interactive")

obs = env.reset()
q = None    # recording keyboard input (in keyboard-control mode)
orn = pos = None    # orn: global angle    pos: global position
rgb = depth = optical_flow = None   # recording sensor information
control_type = "keyboard"   # control mode: 1.random  2.keyboard

def select_action(env, type="random"):
    global q
    if type == "random":
        return env.action_space.sample()
    elif type == "keyboard":
        if q == ord("w"):
            return np.array([0.7, 0.0])
        elif q == ord("a"):
            return np.array([0.0, -0.2])
        elif q == ord("d"):
            return np.array([0.0, 0.2])
        elif q == ord("s"):
            return np.array([-0.7, 0.0])
        elif q == ord(" "):
            return np.array([0.0, 0.0])

for i in range(100):
    if control_type == "keyboard":  # get the input of keyboard   w:up  s:down  a:left  d:right
        while True:
            q = cv.waitKey(1)
            if not q == -1:
                break
    action = select_action(env, control_type)
    obs, reward, done, info = env.step(action)

    # get the information (of the environment) from the sensors (there are many other sensors except for these three)
    # rgb, depth, optical_flow = obs['rgb'], obs['depth'], obs['optical_flow']
    rgb, depth = obs['rgb'], obs['depth']

    # get the GLOBAL position and angle of the robot
    pos_tmp = env.robots[0].get_position()[:2]
    orn_tmp = env.robots[0].get_rpy()[-1:-2:-1]

env.close()
