# Gesture Recognition

### Semester project for Dr. Adam Czajka's Computer Vision class, by Zach Vincent

# Part 3: First update


        A few illustrations demonstrating how your methods processed the training data, for instance show a few segmentation results (2 points).

## Pre-processing and feature extraction
The data for this project is a set of short videos (~1-2 seconds long) that consist of a single ASL gloss (word). Each video is preprocessed according to `preprocess.py`, where each frame is input to MediaPipe. The script outputs a CSV file for each video with 91 features of pose and hand landmarks represented as a fraction of video width and height.

This feature extraction method is ideal for this project because MediaPipe's API is sophisticated and produces reliable landmark outputs. The resulting neural network can be more widely applied since the input data is not dependent on factors such as resolution or height/width ratios. Additionally, it makes the problem solution less of a black box, where feature extraction occurs implicitly somewhere within a convolutional neural network.

> The data used for this solution is currently insufficient. I downloaded and trained my network on [this WLASL dataset](https://dxli94.github.io/WLASL/). The downloading process for this dataset utilizes a Python Python script that remotely downloads a set of thousands of videos included in a JSON file. However, half or more of the videos are hosted on YouTube, which does not support remote video downloading, so all of these videos are skipped. As a result, I have a dataset of hundreds of ASL glosses with only a few examples of each. I was aware of this shortcoming in my **Part 2** submission but I did not realize the extent to which it would ultimately impact performance.

## Setup

The code is now split into two projects: one is my preliminary solution, based just on single-frame hand landmarking. This solution is functional, accurate, and can be run on the webcam with the following:

```
python3 -m pip install -r requirements.txt
python3 recognition.py
```

The second is my newer and more advanced solution, which uses a recurrent neural network to recognize ASL gestures. This solution is technically functional, but due to data collection difficulties (described above), performs around random chance. Setup for this application is identical, but use `translate.py` instead. This uses the webcam.

```
python3 -m pip install -r requirements.txt
python3 translate.py
```

## Future improvements

Now that the proof of concept is complete, I anticipate a number of changes that should improve the model's accuracy and robustness:
- [ ] Decrease the number of glosses included in the training set and increase the number of examples of each gloss.
- [ ] Multithread the preprocessing code. Preprocessing takes a few hours on the CRC machines. Though I'll only need to preprocess an entire dataset once or twice more, this will be an important change for others reusing my code or replicating the results.
- [ ] Normalize the landmarks. The input videos have varying height/width ratios, so two videos showing the same gloss may have very different landmark outputs. A potential solution to this problem could be expressing each landmark's position as a ratio of shoulder width.

# Part 2: Data acquisition and preparation

## Source

I collected data myself for this project using my webcam and the included `get-data.py` file. It is a CSV file with 43 rows. Each row includes 21 hand landmark positions as derived from MediaPipe in (Nx, Ny) format, where N is the landmark number (1-21). MediaPipe outputs these landmarks as fractions of total screen width and height, but to ensure the classifier only considers the relative positions of the landmarks, I normalized the data to be a fraction of the total hand width and height. I believe that this way the classifier works optimally regardless of the position of the hand on the screen. The 43rd row is the gesture, represented currently as `[0, 1, 2]` for `['stop', 'thumbs down', 'thumbs up']`, respectively. My hand data is located in `gesture-data/gesture-data.csv`.

## Train v. Validation

Since MediaPipe extracts the necessary landmarks and I normalize the output, the main source of variability will be in differently proportioned hands. I can retrieve this data from [this dataset](https://www.idiap.ch/webarchives/sites/www.idiap.ch/resource/gestures) or by asking friends to add to my current data. Data preprocessing for the downloaded dataset will require refactoring the code to iteratively extract MediaPipe landmarks from each image and transform into the same CSV format.

## Distinct objects in data

At time of writing, only four distinct gestures exist in the data: stop, thumbs down, thumbs up, and excuse me (single finger pointing upward). The dataset is restricted for the purpose of rapid classification testing. Now that the classification pipeline appears functional, the next step is to add additional data with my left hand (all current data uses my right) and add more gestures. Alternatively, the next step could be the expansion of the project into ASL translation using an RNN (see solution weaknesses).

## Characterization of samples

My laptop webcam is 1280 x 720 pixels and all training was done with sufficient lighting (although more lighting would be beneficial). MediaPipe abstracts away all of the intrinsic image properties, so the processed samples are all continuous float values in the range `[0, 1]` representing the point's Manhattan distance from the origin of the ROI containing the hand (i.e. the topmost landmark will have the highest Y value, the leftmost landmark will have the smallest X value).

## Current solution weaknesses

The gesture recognition is functional. To install and test:

```
python3 -m pip install -r requirements.txt
python3 recognition.py
```

The program will print the recognized gesture to the terminal. This initial solution has several weaknesses:
- [x] Minimal recognition types - only 3 gestures recognized. Solve by adding more gestures to training dataset.
- [ ] At least one gesture is always recognized - classifier (XGBoost decision tree) will always output a classification, even when no gesture is present. Solve by taking training images with gesture label `none` or by changing classifier to a neural network that outputs class probabilities and use a minimum threshold for recognition.
- [ ] Misclassified gestures based on hand orientation - sometimes, stop is interpreted as thumbs up. Solve by including more rotational variety in dataset.
- [x] No ability to recognize gestures that involve movement - only static gestures can be recognized with this classifier. Solve by implementing RNN and use [ASL dataset](https://dxli94.github.io/WLASL/).

    > I have downloaded this dataset. Though I did not download the entire set, I now have on my machine 3071 unique glosses (ASL words) and 21083 total videos, totaling 717MB of unprocessed videos of varying dimensions. Though more data is available, I don't think I'll need it for the purposes of this project. The dataset came with an associated Computational Use of Data document, which is now uploaded to this repository as `C-UDA-1.0.pdf`.

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
