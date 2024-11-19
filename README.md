# Object Tracking

## Introduction

This application is an implementation of People tracking using Tensorflow implementation of MobileNet V3 + SSD[1] and DeepSort[2]. This could very well be extended to any other objects for tracking like cars or animals.

### Approach
1. Find bounding box using MobileNet V3 + SSD of people in the frame
2. Extract features(128 dim) of the people detected in step 1 using a CNN.
3. Estimate trajectory of the person using Kalman Filter.
4. Assign person id :
   - using the features extracted in step 2 and trajectory prediction of the person id using step 3.
   - using a weighted sum of the cosine distance of CNN features and trajectory prediction score.

### Running the tracker
Pass in the relative location of the video file <br>
Example file_dir : ../videos/video_1.avi

```
python object_tracking.py file_dir
```

### Files

- object_tracking.py runs the MobileNet & DeepSORT based people tracking application. 

- object_detection.py runs the object detection using MobileNet+SSD on a video stream.

- ./deep_sort/deep_sort_app.py contains the implementation of the deep sort algorithm[2].

- mobile_net.py contains the implementation of finding bounding boxes of our roi using mobilenet[1]

## References

Pre-trained Mobilenet file at https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1

<a id="1">[1]</a> 
M Sandler (2018). 
MobileNetV2: Inverted Residuals and Linear Bottlenecks [arxiv link](https://arxiv.org/abs/1801.04381)

<a id="1">[2]</a> 
N Wojke (2017). 
Simple Online and Realtime Tracking with a Deep Association Metric [arxiv link](https://arxiv.org/abs/1703.07402)

<a id="1">[3]</a> https://github.com/nwojke/deep_sort

## Requirement
- Python3
- TensorFlow >= 2.1
- OpenCV
