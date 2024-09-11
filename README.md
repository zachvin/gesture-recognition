# Gesture Recognition

### Semester project for Dr. Adam Czajka's Computer Vision class, by Zach Vincent

# Part 1: Conceptual design

## Problem description

The final goal of this project is to have a software product that uses a webcam to recognize a set of predefined hand gestures such as a thumbs up and OK sign. This requires the program to grab a frame from the camera and output a textual description of the shown gesture superimposed on the video preview. The minimum required framerate for this project is 1Hz.

## Task breakdown

1. Get image data from camera
    - This is easily achieved with a loop using OpenCV.

2. Segment hand
    - There are a number of ways to achieve hand segmentation. The first is using Google's MediaPipe, which will automatically label key points on the hand such as fingertip and palm positions.
    - Another method is through using a watershed algorithm to segment the hand from the rest of the image. At this point, the next step is to extract the location of the fingers or to feed this ROI directly into a CNN.
    - It may also be possible to use contour detection to draw a convex hull around the shape of the hand. From here, it is possible to count the number of fingers that are raised by counting the number of points on the outside of the hull. However, this method would not be precise enough to determine the orientation of bent fingers.

3. Classify features
    - This can also be done in any number of ways, divided broadly into classifiers and DNNs. Classifiers like kNN, SVMs, and perhaps decision trees may be effective enough for gesture recognition. For these classifiers, I may restructure the features to be relative to one another as opposed to absolute pixel positions.
    - A DNN should also be effective in recognizing these gestures. However, this method would make it more difficult to extend the number of gestures in the algorithm's dictionary, since the model would have to be retrained to include an additional node on the output layer. Alternatively, a DNN could be used to output a vector of discrete descriptors about the hand, which could then be mapped to an expandable set of gestures.
  
From all of these options, I think the most time-efficient implementation will be using MediaPipe. I could take ~50 images for 3 gestures, input them into MediaPipe, and save the output with the correct label. Then, I would determine a baseline performance with a random forest regressor. I would then test the performance using 20 additional images for each gesture.
  
## Data Acquisition

Depending on the classification approach, it is possible that not much data will be needed. In these cases, it will be straightforward to take a video of a hand in a specific gesture, moving it around the camera. The features are then extracted from each frame and automatically labeled as the selected gesture. With a relatively high framerate, this makes it simple to collect a relatively large amount of data.

With a neural network, it may be required to have significantly more data. Some promising online datasets include [this one](https://www.idiap.ch/webarchives/sites/www.idiap.ch/resource/gestures/) and [this one](http://www-vpu.eps.uam.es/datasets/HGds/).

## Stretch goals

* I would be very interested in exploring a context-dependent neural network, such as an RNN, to recognize gestures that contain information along the time axis, such as sign language. This requires a large dataset, which can be found [here](https://dxli94.github.io/WLASL/).
* Support for multiple hands in the frame at a given time.
* Simple API to export hand gestures for computer control.
