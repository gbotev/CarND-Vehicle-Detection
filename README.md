
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car.png
[image2]: ./output_images/notcar.png
[image3]: ./output_images/rotated_car.png
[image4]: ./output_images/rotated_notcar.png
[image5]: ./output_images/rotated_car_notcar.png
[image6]: ./output_images/detected.png
[image7]: ./output_images/false_positive.png
[image8]: ./output_images/undetected.png
[image9]: ./output_images/no_false_positive.png
[image10]: ./output_images/undetected_detected.png
[video1]: ./project_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I am reusing a large part of the code given in the lectures. Most functions are defined in the first cell of the jupyter notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
![alt text][image2]

All tunable parameters may be found in the fourth cell of the notebook.

To speed up the process I have resized the images to 32x32. I have also tried with 64 by 64 and 16 by 16, but 32 by 32 seemed to be the best choice - with 64 I was overfitting and with 16 - underfitting.

Because the initial count of images is a little bit low (around 8000 cars and 8000 notcars images) I have done some simple data augmentations: I have flipped horizontally the car images to obtain more car images. I have flipped the car and noncar images vertically and then also horizontally to obtain more non car images.  This results in an inbalance of car vs non-car examples of 1 : 3 (cars: 17584 not cars: 53456). This leads to a reduction of the false positives. It also doesn't seem to be worsening the performance of the model by increasing the number of false negatives.

Here are some examples.

For a car:

![alt_text][image3]

For notcar:

![alt text][image4]
![alt text][image5]

For color spaces I have chosen YCrCb as it lead to best results in the project exercises. It is also mentioned a multiple times by the instructors and documentation.

For orientations I have tried 8 and 16. Surprisingly for me 16 leads to better results (I supposed that the model would overfit too much). Though it also leads to twice as high processing time, and the improvement is not so big.

For pixels per cell I have been working with 4 most of the time, but to improve performance I decided to go with 8 pixels per cell, otherwise one frame was processing by more than 3 seconds.

For cells per block I have used 2. I have tried also 1 and three, but 1 leads to very bad model accuracy, although the performance gain is gfreat (around 4 times). With 3 the model was overfitting.

For the simpler features like histograms and spatial features I have tried turning them both on and off. I thought that with them, off the model would work better because it would learn to better use the hog features. This was not the case so I decided to leave both features on with spatial size 16 by 16 and 32 histograms per channel.

I have used linear svm classifier. I have tried different gammas, but I was unable to get a better result than the default value. For the C parameter I have tried values between 0.01 and 1 (0.01, 0.03, 0.1, 0.3, 1). The middle value of 0.1 worked best for me.

The other parameters are simpler and does not affect the model directly although they are related to the final performance. Those are the xy overlaps of windows, resizing of the initial image, x and y start and stop values, search windows sizes and heat threshold.

Overlap was chosen to be 0.5 as it gives best trade off between speed and accuracy (higher overlaps are finding cars better, but also are more CPU demanding).

I have resized the image down to 320 by 180 pixels. Resizing down below that let to poor performance of the classifier.

I have not tuned x and y start and stop parameters - I have used the values I obtained in the lectures for y - 0.555*height to the end and "None" for the x using all the image width.

For the window search size I am using 32 and 40, for heat threshold 0.8 (it is converted to int in the code for heat map). Those parameters are connected and the best performance is obtained by finding a balance between them.

The final tunable parameter is frame memory length - it determines for how many frames each bounding box found is kept in memory and used for finding cars. I have arbitrarily set that to 10.


#### 2. Explain how you settled on your final choice of HOG parameters.

Above I have explained how I tuned the parameters for the HOG templates. I have not done some special HOG tuning image by image. I have just tried out some combinations of parameters and looked at the final model performance, by using just HOG features without the spatial and histograms. It is a bit hard to tune just by using the model performance on the test set, because it is very high (98%-99%) for almost all models I trained. 

Thus I have been using the supplied test images to determine if the parameter is doing a good job or not. Sometimes a model had a 97% score on the test set and failed to find any car on one of the test images, or had 4-5 false positives.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The SVM classifier is trained in the sixth cell of the notebook. I am using a feature vector of length 1728 and the train images count is 56832 (80%). The test images are 14208.

The obtained performance was:

976.49 Seconds to train SVC... Test Accuracy of SVC = 0.9881

I have explained how I tuned the C parameter in point 1. Unfortunately the long time needed to train a model have prevented me to try out more combinations of model and features.

Though I find the current model gives satisfactory results.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I have used the window search code provided in the lectures. The final parameters I use are 0.5 overlap and 32 and 40 pixel windows. 

I have also tried the method with generating a HOG over the whole image and then using those features, but I ran into a problem. Using this method I obtained slightly different windows than using the original window-generating method, although I thought they should be the same. Also it was hard for me to experiment with the overlap values. Finally this method didn't lead to the expected speed ups of 3-4 times when using 50% overlap, ultimately resulting in slightly better (10-15%) performance.

For these reasons and for the simplicity I preferred to use the first window-generator.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  

Here are some example images:

Detected bith cars:

![alt text][image6]

There are some false positives:

![alt text][image7]

Also some cars remain undetected:

![alt text][image8]

Though this could be fixed by tuning the sliding windows parameters. For example:

If the heat threshold is increased the false positives disappear:

![alt text][image9]

If a smaller search window is added (e.g. 12x12 pixels) we can find the undetected cars:

![alt text][image10]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I have implemented the heat function from the lectures for each frame. The pipeline is as follows: for each generated window in the frame we resize it to 32 by 32 and check if this is an image of a car or not using the SVM classifier. Then using the heat function we are adding those bounding boxes on a black image, where the pixels within each box add a value of 1 to the corresponding pixel location. Finally we threshold by using the heat threshold parameter to remove pixels with values lower than it and generate the final bounding boxes given the data from the heat map.

For the video I am using a slightly more elaborated pipeline as follows:

For the first n (hypermater) images we just add the bounding boxes to an array of "windows". While this array increases we also increase the threshold for the heat map by the length of the array - heat_threshold = per_frame_threshold * len(windows_array).

After we have more than n images, we pop out the first (earliest) bounding boxes and append the the recently find ones at the end.

In this way the output is fairly robust to false positives in the current frame, as they should have also been present in the last n frames.

The negative side is that sometimes we lose track of the real cars we are following.

The parameters for the sliding windows size, overlap and heat threshold should be finely tuned in order to obtain best results, though the relatively slow processing time (around 0.5-1 second per frame) is a major drawback.

The result is that in the video we have some false positives, we also have some frames where we lose the cars, or detect two cars as one. The final problem is that the bounding boxes sometimes appear too square (because of the large window size we search) and the position of the car can be exaggerated by about 30-40%.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

First point to consider is lowering the computation times. On one hand a faster machine can be used to speed up the SVM training, so more pictures (e.g. the ones supplied by udacity) can be used for training. Also if we have more data we can use better features without overfitting (e.g. 64x64 car iamges, 16 orientations for HOG).

On the other hand we should speed up the inference process. One way to do that is when processing the video to check just around the bounding boxes found in the last frame and only periodically doing a full check.

A better implementation of the method where we compute the HOG features for the whole image and then speed up the windows search can also be considered.

Also some cases remain to be seen. For example my detector have difficulties tracking white cars on white road and black cars on black road. Probably it would be even harder to follow them when it is dark.

One final point to consider is that the model usually fails when there is some elaborate shadow on the road and it detects cars there. Probably the training "not cars" images are mostly low frequency images, so high frequency images are automatically classified by cars.

Finally I would try how a "modern" deep learning framework would do. Probably it would be superior to hand crafted features and also the training process could be speed up by using GPUs (maybe scikit also can be sped up, though I have no experience there). Also with convolutions the whole image could easily be convolved at the beginning without side effects on the performance of the model, that I experienced using the full image HOG computation.


