import gym
import numpy as np

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

from sb3_contrib import QRDQN

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)

# Wrappers
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecTransposeImage


class NStepReplayBuffer(ReplayBuffer):
    """
    Replay Buffer that computes N-step returns.
    :param buffer_size: (int) Max number of element in the buffer
    :param observation_space: (spaces.Space) Observation space
    :param action_space: (spaces.Space) Action space
    :param device: (Union[th.device, str]) PyTorch device
        to which the values will be converted
    :param n_envs: (int) Number of parallel environments
    :param optimize_memory_usage: (bool) Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
    :param n_steps: (int) The number of transitions to consider when computing n-step returns
    :param gamma:  (float) The discount factor for future rewards.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        device = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        n_steps: int = 1,
        gamma: float = 0.99,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs, optimize_memory_usage)
        self.n_steps = int(n_steps)
        if not 0 < n_steps <= buffer_size:
            raise ValueError("n_steps needs to be strictly smaller than buffer_size, and strictly larger than 0")
        self.gamma = gamma

    def _get_samples(self, batch_inds: np.ndarray, env = None):

        actions = self.actions[batch_inds, 0, :]

        gamma = self.gamma

        # Broadcasting turns 1dim arange matrix to 2 dimensional matrix that contains all
        # the indices, % buffersize keeps us in buffer range
        # indices is a [B x self.n_step ] matrix
        indices = (np.arange(self.n_steps) + batch_inds.reshape(-1, 1)) % self.buffer_size

        # two dim matrix of not dones. If done is true, then subsequent dones are turned to 0
        # using accumulate. This ensures that we don't use invalid transitions
        # not_dones is a [B x n_step] matrix
        not_dones = np.squeeze(1 - self.dones[indices], axis=-1)
        not_dones = np.multiply.accumulate(not_dones, axis=1)
        # vector of the discount factors
        # [n_step] vector
        gammas = gamma ** np.arange(self.n_steps)

        # two dim matrix of rewards for the indices
        # using indices we select the current transition, plus the next n_step ones
        rewards = np.squeeze(self.rewards[indices], axis=-1)
        rewards = self._normalize_reward(rewards, env)

        # TODO(PartiallyTyped): augment the n-step return with entropy term if needed
        # the entropy term is not present in the first step

        # if self.n_steps > 1: # not necessary since we assert 0 < n_steps <= buffer_size

        # # Avoid computing entropy twice for the same observation
        # unique_indices = np.array(list(set(indices[:, 1:].flatten())))

        # # Compute entropy term
        # # TODO: convert to pytorch tensor on the correct device
        # _, log_prob = actor.action_log_prob(observations[unique_indices, :])

        # # Memory inneficient version but fast computation
        # # TODO: only allocate the memory for that array once
        # log_probs = np.zeros((self.buffer_size,))
        # log_probs[unique_indices] = log_prob.flatten()
        # # Add entropy term, only for n-step > 1
        # rewards[:, 1:] = rewards[:, 1:] - ent_coef * log_probs[indices[:, 1:]]

        # we filter through the indices.
        # The immediate indice, i.e. col 0 needs to be 1, so we ensure that it is here using np.ones
        # If the jth transition is terminal, we need to ignore the j+1 but keep the reward of the jth
        # we do this by "shifting" the not_dones one step to the right
        # so a terminal transition has a 1, and the next has a 0
        filt = np.hstack([np.ones((not_dones.shape[0], 1)), not_dones[:, :-1]])

        # We ignore self.pos indice since it points to older transitions.
        # we then accumulate to prevent continuing to the wrong transitions.
        current_episode = np.multiply.accumulate(indices != self.pos, 1)

        # combine the filters
        filt = filt * current_episode

        # discount the rewards
        rewards = (rewards * filt) @ gammas
        rewards = rewards.reshape(len(batch_inds), 1).astype(np.float32)

        # Increments counts how many transitions we need to skip
        # filt always sums up to 1 + k non terminal transitions due to hstack above
        # so we subtract 1.
        increments = np.sum(filt, axis=1).astype(np.int32) - 1

        next_obs_indices = (increments + batch_inds) % self.buffer_size
        obs = self._normalize_obs(self.observations[batch_inds, 0, :], env)
        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(next_obs_indices + 1) % self.buffer_size, 0, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[next_obs_indices, 0, :], env)

        dones = 1.0 - (not_dones[np.arange(len(batch_inds)), increments]).reshape(len(batch_inds), 1)

        data = (obs, actions, next_obs, dones, rewards)
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

# Should be changed to account for multiple environments with different reward span
class NormalizeReward(gym.RewardWrapper):
    def __init__(self,env):
        super(NormalizeReward,self).__init__(env)
        self.env.reward_range = (-1,1)

    def reward(self,reward):
        return reward / 15

env = gym_super_mario_bros.make('SuperMarioBros2-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)
env = NormalizeReward(env)

device = "cuda"
policy_kwargs = {"net_arch" : [512,512,512,512]}
replay_buffer_kwargs  = {"n_steps" : 7}
model = QRDQN("CnnPolicy", env, policy_kwargs=policy_kwargs,replay_buffer_class=NStepReplayBuffer,
learning_rate=1e-4,gradient_steps=1,replay_buffer_kwargs=replay_buffer_kwargs,buffer_size=50000,
device=device,verbose=1,exploration_fraction=1/7,exploration_initial_eps=.99,exploration_final_eps=.1,
batch_size=256,tensorboard_log="./training/basic")

model.learn(7e6)
model.save("./training/qrdqn_basic_mario")
model.save_replay_buffer("./training/qrdqn_basic_mario_buffer")
