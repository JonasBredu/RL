import gym
import offworld_gym
import os
import numpy as np
import matplotlib.pyplot as plt

from offworld_gym.envs.common.channels import Channels
from offworld_gym.envs.common.actions import FourDiscreteMotionActions
from offworld_gym.envs.real.real_env import AlgorithmMode, LearningType
from stable_baselines3 import DQN
import logging
import stable_baselines3

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy


from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import results_plotter
from stable_baselines3 import PPO

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True

def evaluate(model, num_steps=1000):


     episode_rewards=[0.0]
     obs = env.reset()
     for i in range(num_steps):
         action,_states = model.predict(obs)
         obs, reward, done, info = env.step(action)

         episode_rewards[-1] += reward
         if done:
             obs = env.reset()
             episode_rewards.append(0.0)

     mean_100ep_reward = round(np.mean(episode_rewards[-100:]),1)
     print("Mean reward:",mean_100ep_reward, "Num episodes:", len(episode_rewards))

     return mean_100ep_reward
log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)


env = gym.make("OffWorldDockerMonolithDiscreteSim-v0", channel_type=Channels.RGB_ONLY)
time_steps=200000
name="Offworld_DQN4"

env = Monitor(env, log_dir)
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
model = DQN("MlpPolicy", env, gamma = 0.95, learning_rate=1e-3, verbose=0, buffer_size=1000, batch_size=16, exploration_fraction=0.9, exploration_final_eps=0.1, exploration_initial_eps=1.0, train_freq=1)
print(type(callback))
#, exploration_fraction=0.1, exploration_final_eps=0.02, exploration_initial_eps=1.0, train_freq=1
model.learn(total_timesteps=int(time_steps), callback=callback)

results_plotter.plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, name)
plt.savefig(name+'.png')
model.save(name)

model = DQN.load(name)
mean_reward = evaluate(model, num_steps=100)

env.close()
