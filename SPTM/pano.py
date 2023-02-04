from igibson.envs.igibson_env import iGibsonEnv
import yaml
import cv2 as cv
import numpy as np
from copy import deepcopy
import os

config_data = yaml.load(open('turtlebot_nav.yaml', "r"), Loader=yaml.FullLoader)
scene = ['Beechwood_0_int', 'Benevolence_0_int', 'Benevolence_1_int', 'Benevolence_2_int', 'Ihlen_0_int', 'Merom_0_int',
         'Merom_1_int', 'Pomaria_1_int', 'Pomaria_2_int ', 'Rs_int', 'Wainscott_0_int', 'Wainscott_1_int']
scene_id = 'Beechwood_0_int'

env = iGibsonEnv(config_file=deepcopy(config_data), scene_id=scene_id, mode="gui_non_interactive")

# rgb_pano = env.simulator.renderer.get_equi(mode="rgb", use_robot_camera=True)

rgb_pano = None
q = None

def save(episode, step, path="dataset_pano", type="random"):
    global rgb_pano, scene_id

    if not os.path.exists(path):
        os.makedirs(path)
    path_type = os.path.join(path, type)
    if not os.path.exists(path_type):
        os.makedirs(path_type)
    path_type_scene_id = os.path.join(path_type, scene_id)
    if not os.path.exists(path_type_scene_id):
        os.makedirs(path_type_scene_id)

    for dir in ["rgb_pano"]:
        if not os.path.exists(path_type_scene_id + "/" + dir):
            os.makedirs(path_type_scene_id + "/" + dir)

    id = "%03i_%04i" % (episode, step)
    rgb_pano_path = path_type_scene_id + "/rgb_pano/" + id + ".npy"
    np.save(rgb_pano_path, rgb_pano)
    # rgb_pano.tofile(path_type_scene_id + "/rgb_pano/" + id + ".jpg")

    return

def action(env, type="random"):
    global q
    if type == "random":
        return env.action_space.sample()
    elif type == "control":
        if q == ord("w"):
            return np.array([0.7, 0.0])
        elif q == ord("a"):
            return np.array([0, -0.2])
        elif q == ord("d"):
            return np.array([0, 0.2])
        else:
            return np.array([0.0, 0.0])

type = "control"

for episode in range(1):
    p = env.reset()

    # save(episode, 0, type=type)

    for i in range(1, 10):
        act = action(env, type)
        state, reward, done, info = env.step(act)

        rgb_pano = env.simulator.renderer.get_equi(mode="rgb", use_robot_camera=True)

        save(episode, i, type=type)

        q = cv.waitKey(1)
        if q == ord("t"):
            break
        if done == ord("w"):
            break

env.close()