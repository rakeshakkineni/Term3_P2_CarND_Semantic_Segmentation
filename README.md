# Semantic Segmentation 

## **Semantic Segmentation Project**

The goals of this project are :
* Modify the code to identify road pixels using semantic segmentation techniques. 
* Complete the TODO part of the original project. 
* Train the model and provide proof that the trained model is able to label the pixles of the road correctly. 

[//]: # (Image References)

[image1]: ./output/EPOCH Vs LOSS.png "Undistorted"
[image2]: ./output/um_000081.png 
[image3]: ./output/umm_000007.png 
[image4]: ./output/umm_000008.png
[image5]: ./output/uu_000024.png


---
### Writeup / README
Source code provided by "CarND-Semantic-Segmentation" was used as base for this project. 

### Modifications
File main.py was modified to implement this project. All the functions with TODO were completed. Following are the details of the modifications. 

#### load_vgg function modifications: 
Using Tensor Flow saved_model routine saved vgg model is loaded. Each of the layers of the loaded model outputs are saved into saperate tensors and returned for later use. 

#### layers function modifications:
Outputs of "load_vgg" function are used in this function. This function is used for adding upsampling the output of VGG to original image shape.
- Layer7 of VGG model is  2D convoluted instead of fully connected layer as mentioned in the UDACITY classes and output of this 2D convolution is upsampled by 2 times. 
- Output of the above step is added to 2D convoluted VGG Layer4 and upsampled by 2 times.
- Output of the previous step is added to 2D convoluted VGG Layer 3 and upsampled by 8 times to get original input image dimensions.

#### optimize function modifications:
Outputs of "layers" function are used in this function. Softmax and Adamoptmizer are used to compare the model output with expected output to calculate loss. 

#### train_nn function modifications:
This function was modified to call "get_batches_fn" function and feed the output of the function call to the model along with learning rate. 

#### run function modifications: 
This is main function where in the complete training and testing is performed. This function was modified to call all the above listed functions in the following secens. 
- load_vgg
- layers
- optimize
- train_nn 
- helper.save_inference_samples

All the required inputs for the functions to compile were fed. 

### Development Steps 
I have started with code shown in Project Walkthrough video. I have completed the functions taking ques from class and Project Walkthrough. With couple of trials i was able to compile the code without any errors , had struggled to understand how to match dimensions of the skip layers and the upsampled layers. In project walk through it was mentioned that upsampling needs to be done twice with a stride of 2 each time and a final upsampling of 8 , i followed these ques and implemented the model. 

I did not realize the model would be so big and so i tried a few times to run the training on my laptop but every attempt has failed as even with a batch size of '1' model needed a minimum of 3GB+ GPU RAM and my laptop has just 2GB GPU RAM. I have attempted to train the model on CPU's but the training process was really very slow i.e each EPOCH took 45mins to finish. 

I have used Amazon AWS service, trained it using t2.xlarge instance. G2.2x instance is not sufficient as it only has 4GB GPU RAM and it could at the max support a batch_size of 2. 

I have trained the model using following combinations of parameters. 

1st Attempt: With the following combination i could achieve a loss of 0.764. This was not a satisfactory result. 
  EPOCH: 50
  Batch_Size=10
  training_rate = 10e-6

2nd Attempt: With the following combination i could achieve a loss of 0.031. This result matched my expectation. Refer results section for trend of loss over epoches.
  EPOCH: 50
  Batch_Size=5
  training_rate = 0.0009

### Results
- Following picture shows the trend of Loss over multiple epoches
![alt text][image1]

- Following pictures show the final results
![alt text][image2]

![alt text][image3]

![alt text][image4]

![alt text][image5]

### Limitations
- In some cases road is not properly identified especially if the road has a divider in between. Model can be futher improved by introducing some image preprocessing techniques. Due to the limited time i did not explore further. 
