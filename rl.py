import warnings
warnings.filterwarnings("ignore")
import os 
import numpy as np
import copy
import math
import matplotlib.pyplot as plt
from gym_swimmer import SwimmerEnv

env = SwimmerEnv()
env.reset()

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

eval_callback = EvalCallback(env, best_model_save_path='swimmer',
                             log_path='./logs/', 
                             eval_freq=10000,
                             deterministic=True,
                             n_eval_episodes =20)


batch_size = 64 
n_cpu = 1 

model = PPO("MlpPolicy",
            env,
            policy_kwargs=dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])]),
            n_steps=batch_size * 12 // n_cpu,
            batch_size=batch_size,
            n_epochs=10,
            learning_rate=5e-4,
            gamma=0.9,
            verbose=1,
#             render = True,
            tensorboard_log="swimmer/")

model.learn(int(5e6), callback = eval_callback)