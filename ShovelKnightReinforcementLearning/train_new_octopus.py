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


# Might have to do this if I want to compute the gradients for my model
#@torch.no_grad()
def run():
    # Looks for the shovel knight window
    # if it can't find it, just return a region at the top left (useful for debugging other things)
    def find_shovelknight():
        hwnd = ctypes.windll.user32.FindWindowW(0, "Shovel Knight: Shovel of Hope")
        if hwnd:
            rect = ctypes.wintypes.RECT()
            ctypes.windll.user32.GetWindowRect(hwnd, ctypes.pointer(rect))
            bounding_box = {'top': rect.top , 'left': rect.left, 'width': rect.right - rect.left, 'height': rect.bottom - rect.top}
            return bounding_box
        else:
            print("Cannot find Shovel Knight")
            return {'top': 0 , 'left': 0, 'width': 640, 'height': 320}

    #<editor-fold> Config
    config = Config.Config()

    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config.NOISY_NETS = True
    config.USE_PRIORITY_REPLAY = True


    #misc agent variables
    config.GAMMA=0.99
    config.LR=1e-4

    #memory
    config.TARGET_NET_UPDATE_FREQ = 1
    config.EXP_REPLAY_SIZE = 100000
    config.BATCH_SIZE = 32
    config.PRIORITY_ALPHA=0.6
    config.PRIORITY_BETA_START=0.4
    config.PRIORITY_BETA_FRAMES = 100000

    #epsilon variables
    config.SIGMA_INIT=0.4

    #Learning control variables
    config.LEARN_START = 1
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

    # Get the bounding box, create the screenshot tool, and get the cmap
    sk_box = find_shovelknight()
    sct = mss()
    cmap = plt.cm.get_cmap('tab20')

    # Paramters for image size
    BASE_SHAPE = 1280,640
    IMG_SHAPE = 640,320
    CONV_IMG_SHAPE = 640,640

    N_CLASSES = 14

    # Paramters to pass NMS later
    conf_thres = 0.25
    iou_thres = 0.45

    n_episodes = 1000

    # The device to have the model and images on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # The model of the image
    model = attempt_load('./runs/train/exp3/weights/best.pt',map_location=device)
    stride = int(model.stride.max())
    # Get the names of the model
    names = model.module.names if hasattr(model, 'module') else model.names

    # Tell the gpu all images are the same size to speed things up
    cudnn.benchmark = True

    if device.type != 'cpu':
        # Done once
        model(torch.zeros(1, 3, *CONV_IMG_SHAPE).to(device).type_as(next(model.parameters())))

        o = New_Octopus.N_Octopus(config)
        #o.rainbow.load_w("./new_agent/saved_agents/")
        #o.rainbow.load_replay("./new_agent/saved_agents/")

        input("Press enter when game starts to respond")
        died = False

        total_it = 0
        rewards = torch.zeros(n_episodes).to(device)
        ep_lengths = torch.zeros(n_episodes).to(device)
        distances = torch.zeros(n_episodes).to(device)
        losses = torch.zeros(n_episodes).to(device)
        for epoch in range(n_episodes):
            epoch_reward = 0
            prev_health = 8
            prev_state = torch.zeros(601).to(device)
            prev_action = 0

            distance = 0
            loss = 0
            for it in range(1000):
                total_it += 1
                # The screenshot of the area
                sct_img = np.array(sct.grab(sk_box))
                smaller = cv2.resize(sct_img, IMG_SHAPE, interpolation=cv2.INTER_NEAREST)
                new_img = smaller[:,:,:3]

                # Get the health value
                health = 8.
                for cp in [256,260,270,274,285,289,299,303]:
                    if new_img[17,cp,2] <= 180:
                        health -= 1

                prev_health = health

                #<editor-fold> Object Detect
                # Convert the image to something the network can utilize
                img = letterbox(new_img,640,stride=stride,auto=True)[0]
                img = img.transpose((2, 0, 1))[::-1]
                img = np.ascontiguousarray(img)

                # Convert the image to a tensor on the gpu
                # and then make it compatible with the model
                img = torch.from_numpy(img).to(device)
                img = img.float()
                img = img / 255.0
                if len(img.shape) == 3:
                    img = img[None]

                # Feed the image into the model and then do NMS
                pred = model(img,False,False)[0]
                pred = non_max_suppression(pred,conf_thres,iou_thres,None,False,1000)

                # The objects detected in our screen
                values = pred[0]

                #</editor-fold>

                # <editor-fold> Display output
                # Loop over every thing to draw the boxes on whtie background
                # print_img = np.full_like(smaller,255)
                # for row in values:
                #     color = 255 * np.array(cmap(int(row[5])))
                #     c = int(color[2]),int(color[1]),int(color[0])
                #     top_left = (int(row[0]),int(row[1]))
                #     bottom_right = (int(row[2]),int(row[3]))
                #     cv2.rectangle(print_img,top_left,bottom_right,c,2)
                # cv2.putText(print_img,f"Health: {health}",(250,24),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0))
                # print_img = cv2.resize(print_img,BASE_SHAPE,interpolation=cv2.INTER_NEAREST)

                # # The window that shows what we're looking at
                # cv2.imshow("View",print_img)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     cv2.destroyAllWindows()
                #     break
                # </editor-fold>

                t_i = o.transform_input(values,health)

                action = o.rainbow.get_action(t_i)
                reward, dt = o.calculate_reward(prev_state,prev_action,t_i)
                o.take_action(action)

                prev_state = t_i
                prev_action = action

                l = o.rainbow.update(prev_state,prev_action,reward,total_it)

                if l is not None:
                    loss += l.item()
                epoch_reward += reward
                distance += dt

                print(f"Epoch: {epoch:5}\t It: {it:5}\tReward: {reward:.3f}",end='\r')

                if health == 0:
                    died = True
                    break
            rewards[epoch] = epoch_reward
            ep_lengths[epoch] = it + 1
            distances[epoch] = distance
            losses[epoch] = loss / (it + 1)

            if died == False:
                o.reset_sk()
            else:
                while died:
                    # We need to get the image again
                    sct_img = np.array(sct.grab(sk_box))
                    smaller = cv2.resize(sct_img, IMG_SHAPE, interpolation=cv2.INTER_NEAREST)
                    new_img = smaller[:,:,:3]

                    # Get the health value
                    health = 8.
                    for cp in [256,260,270,274,285,289,299,303]:
                        if new_img[17,cp,2] <= 180:
                            health -= 1

                    # If the player recently died
                    if died == True:
                        # And has returned to full health
                        if health == 8:
                            # Not dead, can train again
                            died = False

            o.rainbow.reset_hx()
            o.rainbow.finish_nstep()

            if epoch % 10 == 0:
                o.rainbow.save_w("./new_agent/saved_agents/")
                o.rainbow.save_replay("./new_agent/saved_agents/")
                torch.save(rewards,"./new_agent/saved_agents/reward_tensor.pt")
                torch.save(ep_lengths,"./new_agent/saved_agents/ep_lengths_tensor.pt")
                torch.save(distances,"./new_agent/saved_agents/distances_tensor.pt")
                torch.save(losses,"./new_agent/saved_agents/losses_tensor.pt")

            print(f"Epoch: {epoch:5}     Episode Length: {it:5}     Total Reward: {epoch_reward:.4f}     Distance: {distance:.4f}     Mean Loss: {loss / it:.4f}")


        o.rainbow.save_w("./new_agent/saved_agents/")
        o.rainbow.save_replay("./new_agent/saved_agents/")
        torch.save(rewards,"./new_agent/saved_agents/reward_tensor.pt")

if __name__ == '__main__':
    run()
