import time
import operator
import pickle
import random
import ctypes
import cv2
import os

from mss import mss
from matplotlib import pyplot as plt

import numpy as np
import vgamepad as vg

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

from new_agent import Config
from new_agent import Layers
from new_agent import Replay
from new_agent import Rainbow
from new_agent import New_Octopus

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression
from utils.augmentations import letterbox



#<editor-fold> Config
config = Config.Config()

config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config.NOISY_NETS = True
config.USE_PRIORITY_REPLAY = True


#misc agent variables
config.GAMMA=0.99
config.LR=1e-4

#memory
config.TARGET_NET_UPDATE_FREQ = 1000
config.EXP_REPLAY_SIZE = 100000
config.BATCH_SIZE = 32
config.PRIORITY_ALPHA=0.6
config.PRIORITY_BETA_START=0.4
config.PRIORITY_BETA_FRAMES = 100000

#epsilon variables
config.SIGMA_INIT=0.4

#Learning control variables
config.LEARN_START = 10000
config.MAX_FRAMES=200000
config.UPDATE_FREQ = 1

#Categorical Params
config.ATOMS = 51
config.V_MAX = 10
config.V_MIN = -10

#Multi-step returns
config.N_STEPS = 5

#data logging parameters
config.ACTION_SELECTION_COUNT_FREQUENCY = 1000
#</editor-fold>

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

o = New_Octopus.N_Octopus(config)
o.rainbow.save_w("./new_agent/saved_agents/")
o.rainbow.save_replay("./new_agent/saved_agents/")
