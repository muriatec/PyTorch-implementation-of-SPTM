from igibson.envs.igibson_env import iGibsonEnv
import yaml
import cv2 as cv
import numpy as np
from copy import deepcopy
import os
import math

config_data = yaml.load(open('turtlebot_nav.yaml', "r"), Loader=yaml.FullLoader)
scene = ['Beechwood_0_int', 'Benevolence_0_int', 'Benevolence_1_int', 'Benevolence_2_int', 'Ihlen_0_int', 'Merom_0_int',
         'Merom_1_int', 'Pomaria_1_int', 'Pomaria_2_int ', 'Rs_int', 'Wainscott_0_int', 'Wainscott_1_int']
scene_id = 'Beechwood_0_int'

env = iGibsonEnv(config_file=deepcopy(config_data), scene_id=scene_id, mode="gui_non_interactive")

pano = None
q = None


def save(episode, step, path="dataset_pano", type="random"):
    global pano, pano_ori, pano_op, scene_id

    if not os.path.exists(path):
        os.makedirs(path)
    path_type = os.path.join(path, type)
    if not os.path.exists(path_type):
        os.makedirs(path_type)
    path_type_scene_id = os.path.join(path_type, scene_id)
    if not os.path.exists(path_type_scene_id):
        os.makedirs(path_type_scene_id)

    for dir in ["pano", "pano_ori", "pano_op"]:
        if not os.path.exists(path_type_scene_id + "/" + dir):
            os.makedirs(path_type_scene_id + "/" + dir)

    id = "%03i_%04i" % (episode, step)
    pano_path = path_type_scene_id + "/pano/" + id + ".npy"
    np.save(pano_path, pano)
    pano_ori_path = path_type_scene_id + "/pano_ori/" + id + ".npy"
    np.save(pano_ori_path, pano_ori)
    pano_op_path = path_type_scene_id + "/pano_op/" + id + ".npy"
    np.save(pano_op_path, pano_op)

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


type = "random"

for episode in range(1):
    p = env.reset()

    # save(episode, 0, type=type)

    for i in range(1, 10):
        q = cv.waitKey(1)
        act = action(env, type)
        state, reward, done, info = env.step(act)

        yaw = env.robots[0].get_rpy()[-1:-2:-1]
        pos = env.robots[0].get_position()[:2]

        x = pos[0]
        y = pos[1]
        ori = yaw[0]

        camera_pose = np.array([x, y, 1.2])
        # 设置相机朝向与机器人朝向一致
        view_direction = np.array([1, ori, 0])
        env.simulator.renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])

        pano = env.simulator.renderer.get_equi(mode="rgb", use_robot_camera=False)

        pano = np.array(pano)
        # 截去机器人本体遮挡部分
        pano = pano[:450, :960, :3]
        # print(pano.shape)
        # 设置一个0矩阵，用于存放机器人朝向角
        ori_mat = np.zeros((450, 960))
        # 计算朝向角在矩阵中对应的位置，并将对应列赋为1
        ori_index = round((ori + math.pi) / (2 * math.pi) * 960)
        # print(ori_index)
        ori_mat[:, ori_index] = 1
        ori_mat = ori_mat[:, :, np.newaxis]

        pano_ori = np.concatenate((pano, ori_mat), axis=2)
        # print(pano_ori.shape)

        pano_op = [pano_ori, pos]

        save(episode, i, type=type)

        if q == ord("t"):
            break
        if done == ord("w"):
            break

env.close()