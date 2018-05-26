## Writeup 
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* read the training images and divide them into training and testing data in 80/20 
* all the images(even video frames) are tested to be 0-255 uint8 values so no normalization was needed
* perform a color transformation to YCrCb color space
* extract 3 features, Spatial feature (image resized) , hog features and histogram features for the YCrCb image
* train the classifer and test it
* sliding-window technique was used to detect cars in the images as a test using heat map as a filtring techinque.
* the same pipeline is applied to video stream and output video is saved.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[image8]: ./debug/hog_1.jpg
[image9]: ./debug/hog_0.jpg
[image10]: ./debug/hist_extra5262.png
[image11]: ./debug/hist_image0600.png
[image12]: ./output_images/test1.jpg
[image13]: ./output_images/test3.jpg
[image14]: ./output_images/test4.jpg
[image15]: ./debug/input_159.jpg
[image16]: ./debug/heat_map_159.jpg
[image17]: ./debug/input_292.jpg
[image18]: ./debug/heat_map_292.jpg
[image19]: ./debug/input_662.jpg
[image20]: ./debug/heat_map_662.jpg
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

## Extracted Features
### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in "code/obj_detect_aux.py" line 17 as a function and being called in the same file line 41

After trials it was found that YCrCb gave a better test score for the calssifer meaning that the data in that color spaces gave more seprations .

Parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`) were chosen not so high in order to reduce overfitting and training time and not so low to make the classfier generalize well to the training set.

I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` once for a car sample and one for not car:


![alt text][image8] 
![alt text][image9]

### Histogram feature

all the channels we histograms were samples to 32 bins and concatenated into a single array "code/obj_detect_aux.py" function called "extract_hist_feature" line 10 called in line 34.

sample output

![alt text][image10]
![alt text][image11]

### respamled image.

all the images were resampled to 32\*32 (for each channel) and the resulting vector was added to the features vector.
"code/obj_detect_aux.py" function "extract_spatial_feature" line 21 called in line 44

### normalization 
finally all the feature were scaled induvadily "code/obj_detect_aux.py" lines 74-80

#### 2. training the classifer.

I trained a linear SVM in file "code/train.py" 
all trainings data were extracted lines 22 and 27
stacked toghther in single vector for input and input labels vector line 30-31
data was divided 80/20 for training and testing in line 33-34

final step the model was fitted with 80% of the data (line 40) score was calcuated (line 43) and classifer was "pickled" to avoid retraining every time (line 45)

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

the search window were generated in "code/handle_image_module.py" (lines 58-85) mainly by trial and error and starting from the suggested values in the QA video.

i decided not to go with the HOG smapling once but to recaluate it for every window after cropping because the false postives were already too high with the cropping technique and by sampling the hog it would increase the problem because of the discontinuity at the edge.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on one scale using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image12]
![alt text][image13]
![alt text][image14]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./test_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

a heat  with a reasonably high thresolhd was used to filter false postives reuslts 

### Here are six frames and their corresponding heatmaps:

![alt text][image15]
![alt text][image16]
![alt text][image17]
![alt text][image18]
![alt text][image19]
![alt text][image20]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

the major problem is the rate of false postives that were in the video, even though the test images had no issues at all.
maybe some persistence/averaging filter.time pressure was also a factor because the semster ends in 4 days and something had to be deleviered.

i am also intrested in trying this but with a Yolo network and comapre both results

the fact that the image itself was used is not so good because it will not generlaize well to real life cars, it might have worked on the project because the data were extracted from the video itself 



