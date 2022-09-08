### Description
This project was for my Junior year, and I always wanted to look into reinforcement learning once I heard about machine learning. I wanted a large project to tackle, so I began reading and researching as much as I could into the field. I picked up books and asked professors and faculty a ton of questions. I decided on this project since it seemed like a very interesting problem and it was an environment that had not yet been tackled. Near the end of the project, my professors stated I should try and put together a paper for IEE COG 2022, a conference on games. This was the first time I had written a formal research paper, and ultimately it did not get accepted. However, I plan to still explore more in my direction with the advice given to me by the review board. This is why I began working with Super Mario Bros: The Lost Levels, as it was a proven learnable gym environment.

### Contents
- Ben_van_Oostendorp_Object_Detectors_Reinforcement_Learning_Normal.pdf
    - This is the paper that I wrote and submitted to IEE COG 2022. It goes much more in depth on my approach and process than this readme file does.
    - It does not include any of the newer work done with Super Mario Bros: The Lost Levels nor any of the feedback from the review board.
- new_agent
    - Config.py
        - Configuration information for agents such as epsilon for exploration, learning rate, buffer size for experience replay, etc.
    - Layers.py
        - Different layers the agent would use. Currently it only has a factorised noisy linear layer.
    - New_Octopus.py
        - Contains the agent that will train (N_Octopus class). It has methods for interacting with the game, resetting the game, and calculating the reward
    - Rainbow.py
        - An implementation of [DeepMind's Rainbow](https://arxiv.org/pdf/1710.02298.pdf) agent. Additionally contains the base agents that Rainbow inherits and builds on top of
    - Replay.py
        - Different forms of experience replay buffers, with the main one being Prioritized Experience Replay Memory
- create_new_octopus_weights.py
    - This file was used to restart the agent from fresh with random weights and an empty experience replay
- custom_policy.py
    - An extension of [Stable Baselines 3 Contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) QRDQN agent, featuring noisy layers and a n-step replay buffer
    - Uses the custom Shovel Knight gym environment
- mario_sb3_simple.py
    - A basic version of Stable Baselines 3 Contrib's QRQDQN and a n-step replay buffer to train on Super Mario Bros: The Lost Levels
    - This is a more traditional approach in terms of what the agent sees as a "state" (Nature CNN)
- mario_sb3_yolo.py
    - An extension of the Super Mario Bros: The Lost Levels gym environment where the "state" of the environment is instead bounding boxes of objects created by [YOLOv5](https://github.com/ultralytics/yolov5) instead of an image of the screen
    - This is the approach I was experimenting with and compared with the "simple" version of the environment
- train_new_octopus.py
    - Creates and trains a N_Octopus agent on Shovel Knight, though not as a gym environment