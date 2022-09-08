from typing import Any, Dict, List, Optional, Type

import gym
from gym import Env, spaces
from stable_baselines3.common.policies import BasePolicy, register_policy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
)
from stable_baselines3.common.type_aliases import Schedule
from torch import nn

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)

import numpy as np
import vgamepad as vg
import time
import operator
import pickle
import random
import ctypes
import cv2
import os
import yaml

from mss import mss
from matplotlib import pyplot as plt

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression
from utils.augmentations import letterbox

from sb3_contrib import QRDQN

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma=0.5):
        super(NoisyLinear, self).__init__()

        # Learnable parameters.
        self.mu_W = nn.Parameter(
            torch.FloatTensor(out_features, in_features))
        self.sigma_W = nn.Parameter(
            torch.FloatTensor(out_features, in_features))
        self.mu_bias = nn.Parameter(torch.FloatTensor(out_features))
        self.sigma_bias = nn.Parameter(torch.FloatTensor(out_features))

        # Factorized noise parameters.
        self.register_buffer('eps_p', torch.FloatTensor(in_features))
        self.register_buffer('eps_q', torch.FloatTensor(out_features))

        self.in_features = in_features
        self.out_features = out_features
        self.sigma = sigma

        self.reset()
        self.sample()

    def reset(self):
        bound = 1 / np.sqrt(self.in_features)
        self.mu_W.data.uniform_(-bound, bound)
        self.mu_bias.data.uniform_(-bound, bound)
        self.sigma_W.data.fill_(self.sigma / np.sqrt(self.in_features))
        self.sigma_bias.data.fill_(self.sigma / np.sqrt(self.out_features))

    def f(self, x):
        return x.normal_().sign().mul(x.abs().sqrt())

    def sample(self):
        self.eps_p.copy_(self.f(self.eps_p))
        self.eps_q.copy_(self.f(self.eps_q))

    def forward(self, x):
        if self.training:
            # Do the noise first
            self.sample()
            weight = self.mu_W + self.sigma_W * self.eps_q.ger(self.eps_p)
            bias = self.mu_bias + self.sigma_bias * self.eps_q.clone()
        else:
            weight = self.mu_W
            bias = self.mu_bias

        return F.linear(x, weight, bias)

#<editor-fold> create_mlp_with_noisy
def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    squash_output: bool = False,) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.
    ADDITION: Each of the layers are noisy linear layers
    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [NoisyLinear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(NoisyLinear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(NoisyLinear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules
#</editor-fold>


class QuantileNoisyNetwork(BasePolicy):
    """
    Quantile network for QR-DQN
    :param observation_space: Observation space
    :param action_space: Action space
    :param n_quantiles: Number of quantiles
    :param net_arch: The specification of the network architecture.
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        features_extractor: nn.Module,
        features_dim: int,
        n_quantiles: int = 200,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ):
        super(QuantileNoisyNetwork, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            net_arch = [64, 64]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.features_extractor = features_extractor
        self.features_dim = features_dim
        self.n_quantiles = n_quantiles
        self.normalize_images = normalize_images
        action_dim = self.action_space.n  # number of actions
        quantile_net = create_mlp(self.features_dim, action_dim * self.n_quantiles, self.net_arch, self.activation_fn)
        self.quantile_net = nn.Sequential(*quantile_net)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Predict the quantiles.
        :param obs: Observation
        :return: The estimated quantiles for each action.
        """
        quantiles = self.quantile_net(self.extract_features(obs))
        return quantiles.view(-1, self.n_quantiles, self.action_space.n)

    def _predict(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        q_values = self(observation).mean(dim=1)
        # Greedy action
        action = q_values.argmax(dim=1).reshape(-1)
        return action

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                n_quantiles=self.n_quantiles,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data

class NQRDQNPolicy(BasePolicy):
    """
    Policy class with quantile and target networks for QR-DQN.
    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param n_quantiles: Number of quantiles
    :param net_arch: The specification of the network architecture.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``torch.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        n_quantiles: int = 200,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):

        super(NQRDQNPolicy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = [64, 64]

        self.n_quantiles = n_quantiles
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.normalize_images = normalize_images

        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "n_quantiles": self.n_quantiles,
            "net_arch": self.net_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }

        self.quantile_net, self.quantile_net_target = None, None
        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the network and the optimizer.
        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self.quantile_net = self.make_quantile_net()
        self.quantile_net_target = self.make_quantile_net()
        self.quantile_net_target.load_state_dict(self.quantile_net.state_dict())
        self.quantile_net_target.set_training_mode(False)

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def make_quantile_net(self) -> QuantileNoisyNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return QuantileNoisyNetwork(**net_args).to(self.device)

    def forward(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        return self.quantile_net._predict(obs, deterministic=deterministic)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                n_quantiles=self.net_args["n_quantiles"],
                net_arch=self.net_args["net_arch"],
                activation_fn=self.net_args["activation_fn"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.
        This affects certain modules, such as batch normalisation and dropout.
        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.quantile_net.set_training_mode(mode)
        self.training = mode

class NStepReplayBuffer(ReplayBuffer):
    """
    Replay Buffer that computes N-step returns.
    :param buffer_size: (int) Max number of element in the buffer
    :param observation_space: (spaces.Space) Observation space
    :param action_space: (spaces.Space) Action space
    :param device: (Union[torch.device, str]) PyTorch device
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

class ShovelKnight(Env):
    def __init__(self,max_time_steps=1000,device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super(ShovelKnight,self).__init__()

        # Environment settings
        self.observation_shape = (601,)
        self.input_size = self.observation_shape[0]
        self.observation_space = spaces.Box(0,1,(601,))
        self.action_space = spaces.Discrete(64,)

        self.alive = True
        self.max_time_steps = max_time_steps - 1
        self.time_steps = 0

        self.controller = vg.VX360Gamepad()

        self.sk_bbox = self.find_shovelknight()
        self.device = device

        self.sct = mss()

        # Paramters for image size
        self.BASE_SHAPE = 1280,640
        self.IMG_SHAPE = 640,320
        self.CONV_IMG_SHAPE = 640,640
        # Paramters to pass NMS later
        self.conf_thres = 0.25
        self.iou_thres = 0.4
        # Number of classes
        self.N_CLASSES = 14

        # Yolo model
        self.yolo = attempt_load('./runs/train/exp3/weights/best.pt',map_location=device)
        self.stride = int(self.yolo.stride.max())
        self.yolo_names = self.yolo.module.names if hasattr(self.yolo, 'module') else self.yolo.names
        cudnn.benchmark = True
        # Save for later to compare with
        self.simple_zero = torch.zeros(1)

        self.cmap = plt.cm.get_cmap('tab20')

    def find_shovelknight(self):
        hwnd = ctypes.windll.user32.FindWindowW(0, "Shovel Knight: Shovel of Hope")
        if hwnd:
            rect = ctypes.wintypes.RECT()
            ctypes.windll.user32.GetWindowRect(hwnd, ctypes.pointer(rect))
            bounding_box = {'top': rect.top , 'left': rect.left, 'width': rect.right - rect.left, 'height': rect.bottom - rect.top}
            return bounding_box
        else:
            print("Cannot find Shovel Knight")
            return {'top': 0 , 'left': 0, 'width': 640, 'height': 320}

    def observe(self,mode="object"):
        if mode == "object":
            if self.device != "cpu":
                sct_img = np.array(self.sct.grab(self.sk_bbox))
                smaller = cv2.resize(sct_img, self.IMG_SHAPE, interpolation=cv2.INTER_NEAREST)
                new_img = smaller[:,:,:3]

                health = self.get_health(new_img)

                img = letterbox(new_img,self.CONV_IMG_SHAPE[0],stride=self.stride,auto=True)[0]
                img = img.transpose((2, 0, 1))[::-1]
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(self.device)
                img = img.float()
                img = img / 255.0
                if len(img.shape) == 3:
                    img = img[None]
                pred = self.yolo(img,False,False)[0]
                pred = non_max_suppression(pred,self.conf_thres,self.iou_thres,None,False,1000)

                # The objects detected in our screen
                values = pred[0]

                self.values = values
                self.print_img = np.full_like(smaller,255)
                self.health = health

                next_state = self.transform_input(values,health)

                self.img = new_img

                return next_state.cpu()

    def normalize_state(self,x,img_size=640):
        x = x / img_size
        x[x < 0] = 0

        return x

    def transform_input(self,state,health):

        # Sorts left to right, bottom to top
        b = state[state[:, 1].sort()[1]]
        res = b[b[:, 0].sort()[1]]

        state = res.clone().detach()

        fixed_loc = torch.zeros(self.input_size,device=self.device)

        #<editor-fold> State Augmentation
        knight = state[state[:,5] == 0][:,:4].flatten()
        knight_size = min(len(knight),4 * 4)
        fixed_loc[0:knight_size] = knight[:knight_size]

        beetle = state[state[:,5] == 1][:,:4].flatten()
        beetle_size = min(len(beetle),10 * 4)
        fixed_loc[16:16 + beetle_size] = beetle[:beetle_size]

        checkpoint = state[state[:,5] == 2][:,:4].flatten()
        checkpoint_size = min(len(checkpoint),3 * 4)
        fixed_loc[56:56 + checkpoint_size] = checkpoint[:checkpoint_size]

        ladder = state[state[:,5] == 3][:,:4].flatten()
        ladder_size = min(len(ladder),9 * 4)
        fixed_loc[68:68 + ladder_size] = ladder[:ladder_size]

        ground = state[state[:,5] == 4][:,:4].flatten()
        ground_size = min(len(ground),21 * 4)
        fixed_loc[104:104 + ground_size] = ground[:ground_size]

        spikes = state[state[:,5] == 5][:,:4].flatten()
        spikes_size = min(len(spikes),3 * 4)
        fixed_loc[188:188 + spikes_size] = spikes[:spikes_size]

        sand = state[state[:,5] == 6][:,:4].flatten()
        sand_size = min(len(sand),20 * 4)
        fixed_loc[200:200 + sand_size] = sand[:sand_size]

        dragon = state[state[:,5] == 7][:,:4].flatten()
        dragon_size = min(len(dragon),20 * 4)
        fixed_loc[280:280 + dragon_size] = dragon[:dragon_size]

        bubble = state[state[:,5] == 8][:,:4].flatten()
        bubble_size = min(len(bubble),6 * 4)
        fixed_loc[292:292 + bubble_size] = bubble[:bubble_size]

        skeleton = state[state[:,5] == 9][:,:4].flatten()
        skeleton_size = min(len(skeleton),3 * 4)
        fixed_loc[79:79 + skeleton_size] = skeleton[:skeleton_size]

        d_wall = state[state[:,5] == 10][:,:4].flatten()
        d_wall_size = min(len(d_wall),5 * 4)
        fixed_loc[316:316 + d_wall_size] = d_wall[:d_wall_size]

        slime = state[state[:,5] == 11][:,:4].flatten()
        slime_size = min(len(slime),6 * 4)
        fixed_loc[348:348 + slime_size] = slime[:slime_size]

        whelp = state[state[:,5] == 12][:,:4].flatten()
        whelp_size = min(len(whelp),4 * 4)
        fixed_loc[372:372 + whelp_size] = whelp[:whelp_size]

        b_knight = state[state[:,5] == 13][:,:4].flatten()
        b_knight_size = min(len(b_knight),3 * 4)
        fixed_loc[388:388 + b_knight_size] = b_knight[:b_knight_size]
        #</editor-fold>.
        # Remove info like class/confidence

        augmented_state = self.normalize_state(fixed_loc)
        augmented_state[self.input_size - 1] = health / 8

        return augmented_state

    def get_health(self,new_img):
        health = 8.
        for cp in [256,260,270,274,285,289,299,303]:
            if new_img[17,cp,2] <= 180:
                health -= 1

        return health

    def reset(self):
        # Reset the game to the starting location
        if self.alive:
            self.reset_sk()
        # Agent dead, need to wait for it to stop being dead
        else:
            while self.health != 8:
                self.observe()

        self.alive = True
        self.time_steps = 0
        self.prev_state = None
        self.next_state = self.observe()

        return self.next_state

    # Necessary to see that things are happening
    def render(self,mode="object"):
        if mode == "object":
            # Loop over every thing to draw the boxes on whtie background
            for row in self.values:
                color = 255 * np.array(self.cmap(int(row[5])))
                c = int(color[2]),int(color[1]),int(color[0])
                top_left = (int(row[0]),int(row[1]))
                bottom_right = (int(row[2]),int(row[3]))
                cv2.rectangle(self.print_img,top_left,bottom_right,c,2)
            cv2.putText(self.print_img,f"Health: {self.health}",(250,24),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0))
            print_img = cv2.resize(self.print_img,self.BASE_SHAPE,interpolation=cv2.INTER_NEAREST)

            # The window that shows what we're looking at
            cv2.imshow("View",self.print_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
        elif mode == "img":
            # The window that shows what we're looking at
            cv2.imshow("View",self.img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()

    def step(self,action):
        self.time_steps += 1
        self.take_action(action)
        self.next_state = self.observe()
        r,d = self.calculate_reward(self.prev_state,action,self.next_state)
        reward = r

        done = False
        if self.time_steps == self.max_time_steps:
            done = True
        if self.health == 0:
            done = True
            self.alive = False

        self.prev_state = self.next_state

        return self.next_state, reward, done, {}

    def calculate_reward(self,state_x,action,next_state_x):
        if state_x is None:
            return 0, 0
        # Scalars for percent reward
        a = .3
        b = .7

        health = state_x[600]
        objects = state_x.clone().detach()[:600].reshape(150,4)

        next_health = next_state_x[600]
        next_objects = next_state_x.clone().detach()[:600].reshape(150,4)

        health_loss = next_health - health

        # Dying is really bad, so big negative reward
        if next_health == 0 and health != 0:
            return -1,0


        row = torch.any(objects[:4],dim=1)
        next_row = torch.any(next_objects[:4],dim=1)

        # Knight not detected
        knight_row = objects[:4][row]
        next_knight_row = next_objects[:4][next_row]

        if(knight_row.shape[0] == 0):
            return a * health_loss.item(), 0
        else:
            knight_row = knight_row[0]

        if(next_knight_row.shape[0] == 0):
            return a * health_loss.item(), 0
        else:
            next_knight_row = next_knight_row[0]


        # Knight is centered
        if knight_row[0] > .42 and knight_row[2] < .58:
            grounds_row = torch.any(objects[26:47],dim=1)
            next_grounds_row = torch.any(next_objects[26:47],dim=1)
            # Find where the ground objects are
            grounds = objects[26:47][grounds_row]
            n_grounds = next_objects[26:47][next_grounds_row]

            # No other objects are found, hard to tell movement
            if grounds.shape[0] == 0 or n_grounds.shape[0] == 0:
                if (action & 8) and not (action & 4):
                    distance_traveled = .02
                elif not (action & 8) and not (action & 4):
                    distance_traveled = 0
                else:
                    distance_traveled = -.02

            else:
                # objects sans knight shapes plus classes
                g_spc = torch.zeros(grounds.shape[0],2)
                n_g_spc = torch.zeros(n_grounds.shape[0],2)

                g_spc[:,0] = grounds[:,2] - grounds[:,0]
                g_spc[:,1] = grounds[:,3] - grounds[:,1]

                n_g_spc[:,0] = n_grounds[:,2] - n_grounds[:,0]
                n_g_spc[:,1] = n_grounds[:,3] - n_grounds[:,1]

                object_1 = -1
                object_2 = -1
                for ob1 in range(g_spc.shape[0]):
                    for ob2 in range(n_g_spc.shape[0]):
                        # Check if the objects are of the same size and in similar positions
                        if torch.all(torch.isclose(g_spc[ob1],n_g_spc[ob2],.02,.02)) and torch.all(torch.isclose(grounds[ob1],n_grounds[ob2],.05,.05)):
                            object_1 = ob1
                            object_2 = ob2

                # No object was capable of satisfying the requirements, too hard to tell movement
                if object_1 == -1 or object_2 == -1:
                    if (action & 8) and not (action & 4):
                        distance_traveled = .01
                    else:
                        distance_traveled = -.01

                distance_r = grounds[object_1][2] - n_grounds[object_2][2]
                distance_l = grounds[object_1][0] - n_grounds[object_2][0]
                distance_traveled = (distance_r + distance_l) / 2
        # Knight is not centered
        else:
            distance_r = next_knight_row[2] - knight_row[2]
            distance_l = next_knight_row[0] - knight_row[0]
            distance_traveled = (distance_r + distance_l) / 2

            # If the screen is transitioning it's moving too fast
            # negate the distance traveled since the knight is "moving forwards"
            # even though it's moving backwards on the screen
            if knight_row[2] - next_knight_row[2] >= .04:
                distance_traveled = -distance_traveled
            elif next_knight_row[2] - knight_row[2] >= .04:
                distance_traveled = -distance_traveled

        # Calculate the reward with scalars
        reward = health_loss * a + distance_traveled * b

        # Return 0 if the reward is really small (potential inconistencies with objects)
        if torch.isclose(reward,self.simple_zero,atol=1e-3):
            return 0,distance_traveled
        else:
            return reward.item(), distance_traveled

    def reset_sk(self):
        self.controller.reset()
        self.controller.update()

        self.controller.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_START)
        self.controller.update()

        time.sleep(1)
        self.controller.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_START)
        self.controller.update()
        self.controller.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP)
        self.controller.update()

        time.sleep(.2)
        self.controller.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP)
        self.controller.update()
        self.controller.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP)
        self.controller.update()

        time.sleep(.2)
        self.controller.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP)
        self.controller.update()
        self.controller.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self.controller.update()

        time.sleep(.2)
        self.controller.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self.controller.update()
        self.controller.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN)
        self.controller.update()

        time.sleep(.2)
        self.controller.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN)
        self.controller.update()
        self.controller.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self.controller.update()

        time.sleep(4)
        self.controller.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self.controller.update()
        time.sleep(.2)
        self.controller.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self.controller.update()

        time.sleep(4)
        self.controller.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self.controller.update()
        time.sleep(.2)
        self.controller.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self.controller.update()

        time.sleep(1)
        self.controller.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self.controller.update()
        time.sleep(.2)
        self.controller.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self.controller.update()

        time.sleep(3)
        self.controller.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self.controller.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_START)
        self.controller.update()
        time.sleep(.1)
        self.controller.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_START)
        self.controller.update()

        time.sleep(.1)
        self.controller.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_START)
        self.controller.update()
        time.sleep(.1)
        self.controller.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_START)
        self.controller.update()

        time.sleep(.1)
        self.controller.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_START)
        self.controller.update()
        time.sleep(.1)
        self.controller.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_START)
        self.controller.update()

        time.sleep(.1)
        self.controller.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_START)
        self.controller.update()
        time.sleep(.1)
        self.controller.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_START)
        self.controller.update()

        self.controller.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_START)
        self.controller.update()

        time.sleep(6)

    def take_action(self,action):
        if(action & 1):
            self.controller.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP)
        else:
            self.controller.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP)
        if(action & 2):
            self.controller.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN)
        else:
            self.controller.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN)
        if(action & 4):
            self.controller.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT)
        else:
            self.controller.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT)
        if(action & 8):
            self.controller.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT)
        else:
            self.controller.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT)
        if(action & 16):
            self.controller.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        else:
            self.controller.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        if(action & 32):
            self.controller.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
        else:
            self.controller.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_X)

        self.controller.update()

    def close(self):
        pass

env = ShovelKnight(1000)
input("Press enter when the game begins to respond")

device = "cuda"
policy_kwargs = {"net_arch" : [512,512,512,512]}
replay_buffer_kwargs  = {"n_steps" : 7}
model = QRDQN(NQRDQNPolicy, env, policy_kwargs=policy_kwargs,replay_buffer_class=NStepReplayBuffer,
learning_rate=1e-4,gradient_steps=1,replay_buffer_kwargs=replay_buffer_kwargs,buffer_size=50000,
device=device,verbose=1,exploration_fraction=0,exploration_initial_eps=0,exploration_final_eps=0,
batch_size=256,tensorboard_log="./training/shovel_knight_sb3_thegucci",learning_starts=1000)

model.learn(10000)
