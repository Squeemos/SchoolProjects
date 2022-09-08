import gym
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression
from utils.augmentations import letterbox

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

class MarioObjectDetection(gym.ObservationWrapper):
    def __init__(self,env,device):
        super(MarioObjectDetection,self).__init__(env)
        #Box(0, 255, (240, 256, 3), uint8)
        self.observation_space = gym.spaces.Box(0,1,(800,))

        # Paramters for image size
        self.BASE_SHAPE = 320,320
        self.IMG_SHAPE = 320,320

        # Paramters to pass NMS later
        self.conf_thres = 0.25
        self.iou_thres = 0.45

        self.N_CLASSES = 16

        self._device_ = device

        self.yolo = attempt_load('./runs/train/exp11/weights/best.pt',map_location=device)
        self.stride = int(self.yolo.stride.max())
        self.yolo_names = self.yolo.module.names if hasattr(self.yolo, 'module') else self.yolo.names
        cudnn.benchmark = True

    def observation(self,obs):
        if self._device_ != "cpu":
            img = cv2.cvtColor(np.array(obs),cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, self.IMG_SHAPE, interpolation=cv2.INTER_NEAREST)
            img = img[:,:,:3]
            img = img.transpose((2, 0, 1))[::-1]
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device)
            img = img.float()
            img = img / 255.0
            if len(img.shape) == 3:
                img = img[None]
            pred = self.yolo(img,False,False)[0]
            pred = non_max_suppression(pred,self.conf_thres,self.iou_thres,None,False,1000)
            values = pred[0]
            return self.transform_input(values).cpu()
        else:
            print("Not using gpu, just returning observation")
            return obs

    def transform_input(self,state):

        # Sorts left to right, bottom to top
        b = state[state[:, 1].sort()[1]]
        res = b[b[:, 0].sort()[1]]

        state = res.clone().detach()

        fixed_loc = torch.zeros(800,device=self._device_)

        #<editor-fold> State Augmentation
        mario = state[state[:,5] == 0][:,:4].flatten()
        mario_size = min(len(mario),4 * 1)
        fixed_loc[0:mario_size] = mario[:mario_size]

        bb = state[state[:,5] == 1][:,:4].flatten()
        bb_size = min(len(bb),4 * 15)
        fixed_loc[4:4+bb_size] = bb[:bb_size]

        star = state[state[:,5] == 2][:,:4].flatten()
        star_size = min(len(star),4 * 2)
        fixed_loc[64:64+star_size] = star[:star_size]

        ground = state[state[:,5] == 3][:,:4].flatten()
        ground_size = min(len(ground),4 * 6)
        fixed_loc[72:72+ground_size] = ground[:ground_size]

        gkoopa = state[state[:,5] == 4][:,:4].flatten()
        gkoopa_size = min(len(gkoopa),4 * 10)
        fixed_loc[96:96+gkoopa_size] = gkoopa[:gkoopa_size]

        block = state[state[:,5] == 5][:,:4].flatten()
        block_size = min(len(block),4 * 50)
        fixed_loc[136:136+block_size] = block[:block_size]

        goomba = state[state[:,5] == 6][:,:4].flatten()
        goomba_size = min(len(goomba),4 * 10)
        fixed_loc[336:336+goomba_size] = goomba[:goomba_size]

        rkoopa = state[state[:,5] == 7][:,:4].flatten()
        rkoopa_size = min(len(rkoopa),4 * 10)
        fixed_loc[376:376+rkoopa_size] = rkoopa[:rkoopa_size]

        fkoopa = state[state[:,5] == 8][:,:4].flatten()
        fkoopa_size = min(len(fkoopa),4 * 10)
        fixed_loc[416:416+fkoopa_size] = fkoopa[:fkoopa_size]

        gshell = state[state[:,5] == 9][:,:4].flatten()
        gshell_size = min(len(gshell),4 * 10)
        fixed_loc[456:456+gshell_size] = gshell[:gshell_size]

        rshell = state[state[:,5] == 10][:,:4].flatten()
        rshell_size = min(len(rshell),4 * 10)
        fixed_loc[496:496+rshell_size] = rshell[:rshell_size]

        iblock = state[state[:,5] == 11][:,:4].flatten()
        iblock_size = min(len(iblock),4 * 20)
        fixed_loc[536:536+iblock_size] = iblock[:iblock_size]

        pipe = state[state[:,5] == 12][:,:4].flatten()
        pipe_size = min(len(pipe),4 * 10)
        fixed_loc[616:616+pipe_size] = pipe[:pipe_size]

        pplant = state[state[:,5] == 13][:,:4].flatten()
        pplant_size = min(len(pplant),4 * 10)
        fixed_loc[656:656+pplant_size] = pplant[:pplant_size]

        flag = state[state[:,5] == 14][:,:4].flatten()
        flag_size = min(len(flag),4 * 2)
        fixed_loc[696:696+flag_size] = flag[:flag_size]

        eblock = state[state[:,5] == 15][:,:4].flatten()
        eblock_size = min(len(eblock),4 * 20)
        fixed_loc[704:704+eblock_size] = eblock[:eblock_size]
        #</editor-fold>.
        # Remove info like class/confidence

        augmented_state = self.normalize_state(fixed_loc)

        return augmented_state

    def normalize_state(self,x,img_size=320):
        x = x / img_size
        x[x < 0] = 0
        x[x > 1] = 1

        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = gym_super_mario_bros.make('SuperMarioBros2-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)
env = NormalizeReward(env)
env = MarioObjectDetection(env,device)

device = "cuda"
policy_kwargs = {"net_arch" : [512,512,512,512,512]}
replay_buffer_kwargs  = {"n_steps" : 7}
model = QRDQN("MlpPolicy", env, policy_kwargs=policy_kwargs,replay_buffer_class=NStepReplayBuffer,
learning_rate=1e-4,gradient_steps=1,replay_buffer_kwargs=replay_buffer_kwargs,buffer_size=50000,
device=device,verbose=1,exploration_fraction=1/7,exploration_initial_eps=.99,exploration_final_eps=.1,
batch_size=256,tensorboard_log="./training/object")

model.learn(7e6)
model.save("./training/qrdqn_object_mario")
model.save_replay_buffer("./training/qrdqn_object_mario_buffer")
