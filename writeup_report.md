# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/center.jpg "Center Lane Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/1968/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolutional neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
It might however be necessary to update the tensorflow version from the provided 1.3 to 1.4, as the pre-trained models provided by the installed keras version are not compatible with the older tensorflow version. Simply run the following in the provided workspace:
```sh
pip install --upgrade tensorflow-gpu==1.4
```

#### 3. Submission code is usable and readable

The model.py file contains the code for preprocessing, training and saving the convolutional neural network. The file shows the pipeline I used for training and validating the model, and it contains comments wherever the code is not self-explanatory.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

For this task I have employed a modified MobileNet architecture, pre-trained on imagenet. The model takes as input images of size (160, 320, 3), then crops away landscape and hood of the car, reducing the image to shape (74, 320, 3), which is then resized to the (128, 128, 3) expected by MobileNet and finally normalized to contain grayscale values in the range [-1,1].

Only the base layers of the pre-trained MobileNet were used (`include_top=False`). The network was then extended by adding a global average pooling layer, a dense layer of 128 neurons with relu activation and finally the output layer with 1 neuron and linear activation function.

#### 2. Attempts to reduce overfitting in the model

The model complexity was chosen as to find the optimal generalizability of the model. Primarily the model complexity can be adapted by changing the `alpha` parameter of the MobileNet, which proportionally changes the number of filters in each Conv2D layer of the architecture. However, I found that the default `alpha=1.0` works well.

I have also employed dropout after each layer of the model head to increase generalizability.

To determine the amount of over-/underfitting for each evaluated architecture, I split the data into train and test dataset, providing a seperate generator for each, and kept track of both learning curves during training. For the chosen architecture both learning curves matched very closely, yielding a final MSE training error of 0.0014 and validation error of 0.00089.

#### 3. Model parameter tuning

The model complexity hyperparameters were already discussed in the previous section.

For simplicity the Adam optimizer was chosen, which avoids the effort of setting a learning rate schedule.

#### 4. Appropriate training data

The training data was collected by driving the car somewhat closely to the center line for 2 full laps. To obtain more precise labels, the car was steered using the mouse, rather than the binary A/D buttons. Due to the response lag and limited granularity of possible steering angles, it was difficult to drive the car at higher speeds.

All 3 camera feeds were used as training data. Adding a steering angle of 0.1 (slight right) to the labels of the left camera images and subtracting 0.1 (slight left) from the right camera images enabled the car to recover after deviating from the center line. Values significantly below 0.1 lost this effect, while larger values led to an unsmooth, oscillating driving behavior.

### Model Architecture and Training Documentation

#### 1. Solution Design Approach

The choice of architecture and tuning of model complexity is already discussed above.

The first version of the model was trained with only center camera images, yielding a good validation MSE. However, it was unable to handle sharp corners in the simulation. Augmenting the data with the left and right camera images (as discussed above) led to the submitted model, which performs well on the track.

#### 2. Final Model Architecture

As discussed above, the architecture consists of pre-processing layers, the MobileNet base layers and a head of 2 dense layers.

MobileNet is a great trade-off between model complexity and model performance. At `alpha=1.0` it yields an imagenet performance similar to architectures such as GoogleNet and VGG16, while having an order of magnitude fewer weights. It achieves this by employing depthwise separable convolutions (see [original paper](https://arxiv.org/pdf/1704.04861.pdf) for more details), which reduce the number of weights and FLOPS by a factor of ~10 for common filter sizes. 

Hence, MobileNet is ideal for transfer learning problems such as this with only few images to train on. In a car, the significantly lower inference time would also be beneficial, as the model needs to run with >10fps on the cheapest, most low-power device possible.

#### 3. Creation of the Training Set & Training Process

As described above, I recorded 2 laps of center lane driving. A sample image is provided below:

![alt text][image1]

This dataset of 3x5334 images was sufficient to get good driving behavior using the above model.

To further improve the performance, recordings from recovering back to the center could have been useful, as well as driving the track in the opposite direction. Augmenting the dataset with flipped images would also be a simple step to double the amount of diverse training data.

The training and validation dataset were read from disk on demand using a generator, which provides 1 batch of images and the corresponding labels on every call. The generator also shuffles the data, once before selecting images of a batch and then once more within the batch.

### Simulation

#### 1. Navigation using the trained model

The model provided as `model.h5` performed as shown in `video.mp4`. At no time during the full lap does a tire leave the track. The car is not always perfectly in the center of the lane but always recovers in a smooth and non-oscillating manner.