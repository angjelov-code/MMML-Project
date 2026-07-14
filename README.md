# Final Project for Modern Methods in Machine Learning

An implementation of a Variational Autoencoder(VAE) on the Celeb A dataset.
Images were generated from random latent vectors, 
reconstruction from test data and
linear interpolation between two images

## Model Architecture

The VAE encoder consists of:
1. Encoder with three convolutional and two dense layers
2. Latent space with dimension 128
3. Decoder with two dense and three up-scaling layers.

## Training

The model was trained on the entire Celeb A dataset, specifically the Align&Cropped Images which can be found [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) .

## The Results

### Example of Generation

![image](images/Generation.png)

### Example of Reconstruction

![image](images/Reconstruction.png)

### Example of Interpolation

![image](images/Interpolation.png)
