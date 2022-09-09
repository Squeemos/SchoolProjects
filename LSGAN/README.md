### Contents
- Final_Project.pdf
  - This is a writeup of the implementation of the Least Squares Generative Adversarial Network (LSGAN)
  - It includes some of the results I achieved after training the LSGAN on the cat face dataset (referenced on the main page)
- LSGAN.py
  - Includes the implementation of the LSGAN
  - It is broken up into two parts:
    - Generator
      - A generator that uses upsamble blocks, convolution blocks, and LeakyReLU blocks to generate an image from a "recipe" (random noise)
    - Discriminator
      - It is basically a convolutional neural network that looks to classify images
      - However, in the LSGAN, the discriminator doesn't give outputs between 0 and 1, it instead can be almost any number since the least squares error (LSE) will minimize the error over time
  - The cat faces are then loaded into a pytorch dataset for better and faster iterations, a model is then made and trained
  
