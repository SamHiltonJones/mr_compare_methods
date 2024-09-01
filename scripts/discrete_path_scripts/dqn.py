import os
import numpy as np
from datetime import datetime
from stable_baselines3 import DQN
import torch as th
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from typing import Callable
from gym import spaces
import gym

class TupleToBoxWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(TupleToBoxWrapper, self).__init__(env)
        
        self.observation_space = spaces.Box(
            low=np.concatenate([space.low for space in env.observation_space.spaces]),
            high=np.concatenate([space.high for space in env.observation_space.spaces]),
            dtype=np.float32
        )

    def observation(self, observation):
        return np.concatenate(observation)

def create_env(env_path, worker_id=0, time_scale=2.0, no_graphics=False):
    channel = EngineConfigurationChannel()
    unity_env = UnityEnvironment(env_path, side_channels=[channel], worker_id=22, no_graphics=no_graphics, base_port=11)
    channel.set_configuration_parameters(time_scale=time_scale)
    
    gym_env = UnityToGymWrapper(unity_env, allow_multiple_obs=True)
    gym_env = TupleToBoxWrapper(gym_env)
    
    return gym_env

def get_save_path(base_path, model_name, trained_path=False):
    date_str = datetime.now().strftime("%d%m%Y")
    version = 0
    if trained_path:
        save_path = f"{base_path}/{model_name}_{date_str}_v{version}.zip"
        while os.path.exists(save_path):
            version += 1
            save_path = f"{base_path}/{model_name}_{date_str}_v{version}.zip"
    else:
        save_path = f"{base_path}/{model_name}_{date_str}_v{version}"
        while os.path.exists(save_path):
            version += 1
            save_path = f"{base_path}/{model_name}_{date_str}_v{version}"
    return save_path[:-4] if trained_path else save_path

class RewardLoggingCallback(BaseCallback):
    def __init__(self, log_interval, base_path, model_name, verbose=0):
        super(RewardLoggingCallback, self).__init__(verbose)
        self.log_interval = log_interval
        self.base_path = base_path
        self.model_name = model_name
        self.step_count = 0
        self.total_reward = 0

        self.log_file = self.get_log_file_path()

    def get_log_file_path(self):
        save_path = get_save_path(self.base_path, self.model_name)
        os.makedirs(save_path, exist_ok=True)
        log_file = f"{save_path}/{self.model_name}_reward_log.txt"
        version = 0
        while os.path.exists(log_file):
            version += 1
            log_file = f"{save_path}/{self.model_name}_reward_log_v{version}.txt"
        return log_file

    def _on_step(self) -> bool:
        self.step_count += 1
        self.total_reward += np.sum(self.locals['rewards'])

        if self.step_count % self.log_interval == 0:
            with open(self.log_file, 'a') as f:
                f.write(f'Step: {self.step_count}, Reward: {self.total_reward}\n')
            print(f'Logged reward at step {self.step_count}: {self.total_reward}')
            self.total_reward = 0
        return True

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

if __name__ == '__main__':
    n_envs = 1

    env = r"builds\discrete_path\3DPos.exe"
    def make_env_scene1(worker_id):
        return lambda: create_env(env, worker_id, time_scale=4.0, no_graphics=True)

    env_fns_scene1 = [make_env_scene1(i) for i in range(n_envs)]
    env_scene1 = SubprocVecEnv(env_fns_scene1)

    base_path = 'discrete_path_models/logs_models'
    trained_models_path = 'discrete_path_models/trained_models'
    os.makedirs(trained_models_path, exist_ok=True)

    model_name = 'DQN'
    save_path_scene1 = get_save_path(base_path, model_name)
    checkpoint_callback_scene1 = CheckpointCallback(save_freq=250000, save_path=save_path_scene1, name_prefix=model_name)

    policy_kwargs = dict(activation_fn=th.nn.ReLU,
                         net_arch=[128, 256, 128])

    tensorboard_log_path_scene1 = get_save_path("./logs_graphs", model_name)
    reward_logging_callback_scene1 = RewardLoggingCallback(log_interval=1000, base_path=base_path, model_name=model_name)

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = DQN("MlpPolicy", env_scene1, verbose=2, tensorboard_log=tensorboard_log_path_scene1,
                policy_kwargs=policy_kwargs,
                device=device)

    model.learn(total_timesteps=2000000, reset_num_timesteps=True, tb_log_name="train_scene3",
                callback=[checkpoint_callback_scene1, reward_logging_callback_scene1])

    final_model_path = get_save_path(trained_models_path, model_name, trained_path=True)
    model.save(final_model_path)
    env_scene1.close()
