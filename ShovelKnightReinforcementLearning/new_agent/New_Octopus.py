import time
import operator
import pickle
import random

import numpy as np
import vgamepad as vg

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

from . import Config
from .Layers import NoisyLinear as NoisyLinear
from .Replay import PrioritizedReplayMemory
from .Rainbow import Rainbow

class N_Octopus(object):
    def __init__(self,config,input_size=601):
        self.controller = vg.VX360Gamepad()
        self.rainbow = Rainbow(input_shape=torch.Size([601]),config=config)

        self.input_size = input_size
        self.device = config.device

        # So when we compare to 0, we don't have to create a new tensor every time
        self.simple_zero = torch.zeros(1).to(self.device)

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

    # Resets shovel knight to the current area
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

    # This function needs a lot of help
    def calculate_reward(self,state_x,action,next_state_x):
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
            return -1


        row = torch.any(objects[:4],dim=1)
        next_row = torch.any(next_objects[:4],dim=1)

        # Knight not detected
        knight_row = objects[:4][row]
        next_knight_row = next_objects[:4][next_row]

        if(knight_row.shape[0] == 0):
            return a * health_loss, 0
        else:
            knight_row = knight_row[0]

        if(next_knight_row.shape[0] == 0):
            return a * health_loss, 0
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
        if torch.isclose(reward,self.simple_zero):
            return 0,distance_traveled
        else:
            return reward.item(), distance_traveled

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
