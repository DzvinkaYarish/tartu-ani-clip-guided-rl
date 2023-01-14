import gym
import gym.utils.play
from stable_baselines3 import DQN
import open_clip
from PIL import Image
import torch
import numpy as np
from tensorboardX import SummaryWriter
import os
import scipy.stats as stats
import json
import matplotlib.pyplot as plt
import uuid

import utils


# ENV_NAME = 'LunarLander-v2'
# env = gym.make(ENV_NAME)
# env.reset()
# img = env.render(mode="rgb_array")
#
# plt.imshow(img)
# plt.show()
# env.close()

env = gym.make('LunarLander-v2')
env = gym.wrappers.RecordVideo(
    env,
    "C:/Users/IronTony/Projects/Python/tartu-ani-clip-guided-rl/videos/",
    episode_trigger=lambda x: True,
    name_prefix=str(uuid.uuid4())
)
gym.utils.play.play(env, keys_to_action={(ord('a'),): 0, (ord("r"),): 1, (ord("e"),): 2, (ord("w"),): 3})
env.close()
