import random
import cv2 as cv
import numpy as np
from igibson.envs.igibson_env import iGibsonEnv
import torch.optim as optim
import resnet
import torch.utils.data as data
import torch
import torch.nn as nn
from train_setup import *


EDGE_EPISODES = 1
MAX_CONTINUOUS_PLAY = 100
MAX_ACTION_DISTANCE = 5
NEGATIVE_SAMPLE_MULTIPLIER = 5
BATCH_SIZE = 2
EDGE_CLASSES = 2
LEARNING_RATE = 1e-04
EDGE_EPOCHS = 10


def data_generator():
    env = iGibsonEnv(config_file=config_data, scene_id=scene_id, mode="gui_non_interactive")
    x_result = []
    y_result = []
    for episode in range(EDGE_EPISODES):
        x = []
        state = env.reset()
        current_x = state['rgb']
        # x.append(current_x)
        for i in range(MAX_CONTINUOUS_PLAY):
            act = env.action_space.sample()
            state, reward, done, info = env.step(act)
            current_x = state['rgb']
            x.append(current_x)
        first_second_label = []
        current_first = 0
        while True:
            y = None
            current_second = None
            if random.random() < 0.5:
                y = 1
                second = current_first + random.randint(1, MAX_ACTION_DISTANCE)
                if second >= MAX_CONTINUOUS_PLAY:
                    break
                current_second = second
            else:
                y = 0
                second = current_first + random.randint(1, MAX_ACTION_DISTANCE)
                if second >= MAX_CONTINUOUS_PLAY:
                    break
                current_second_before = None
                current_second_after = None
                index_before_max = current_first - NEGATIVE_SAMPLE_MULTIPLIER * MAX_ACTION_DISTANCE
                index_after_min = current_first + NEGATIVE_SAMPLE_MULTIPLIER * MAX_ACTION_DISTANCE
                if index_before_max >= 0:
                    current_second_before = random.randint(0, index_before_max)
                if index_after_min < MAX_CONTINUOUS_PLAY:
                    current_second_after = random.randint(index_after_min, MAX_CONTINUOUS_PLAY - 1)
                if current_second_before is None:
                    current_second = current_second_after
                elif current_second_after is None:
                    current_second = current_second_before
                else:
                    if random.random() < 0.5:
                        current_second = current_second_before
                    else:
                        current_second = current_second_after
            first_second_label.append((current_first, current_second, y))
            current_first = second + 1
        random.shuffle(first_second_label)
        for first, second, y in first_second_label:
            # print(second)
            future_x = x[second]
            current_x = x[first]
            current_y = y
            x_result.append(np.concatenate((current_x, future_x), axis=2))
            y_result.append(current_y)
    x_result = np.array(x_result)
    y_result = to_categorical(y_result, num_classes=EDGE_CLASSES)
    # print(x_result.shape, y_result.shape)
    env.close()
    return x_result, y_result


if __name__ == '__main__':

    ## load dataset and data processing
    # raw_x_size: batch size * height * width * (columns * 2), y_size = batch size * 2
    training_data_x, training_data_y = data_generator()

    # Conv2d input size: batch size * columns * height * width
    training_data_x = training_data_x.transpose((0, 3, 1, 2))

    training_data_x = torch.Tensor(training_data_x)
    training_data_y = torch.Tensor(training_data_y)
    print(training_data_x.shape, training_data_y.shape)
    # test_1 = training_data_x[0, :, :, :].reshape(1, training_data_x.shape[1], training_data_x.shape[2], training_data_x.shape[3])
    # print(test_1.shape)
    training_data = data.TensorDataset(training_data_x, training_data_y)

    loader = data.DataLoader(dataset=training_data, batch_size=BATCH_SIZE, shuffle=True)

    ## network: Resnet18-based siamese
    model = resnet.generate_model(18)

    ## optimizer: Adam
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    ## loss function
    loss_fn = nn.CrossEntropyLoss()

    # test_1 = torch.Tensor(test_1)
    # print(test_1.shape)
    # output = model(test_1)
    # print(output)

    for i in range(EDGE_EPOCHS):
        model.train()
        for step, [inputs, labels] in enumerate(loader):
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = loss_fn(outputs, labels)
            loss.backward()

            optimizer.step()
