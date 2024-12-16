# Gesture Recognition

### Semester project for Dr. Adam Czajka's Computer Vision class, by Zach Vincent

# Part 5: Final update

## Database description

The data that I use for this project is a word-level ASL dataset from a research paper. The dataset is downloaded via a script that collects and downloads videos from various websites, including YouTube. The data that I downloaded was about 20,000 videos representing about 3,000 ASL words ("glosses"). For my training, I used a subset of 39 of the most frequent glosses in the training data. There is an unequal number of instances of each gloss in the dataset, so to maximize training efficiency, I removed the less frequent glosses from the training data. The training and validation subsets are randomly selected from this subset. I believe that random selection for the validation subset is sufficient because each video is slightly different in aspect ratio, signer, signing speed, and exact signing motion. In this way, each sample for a specific gloss varies except for the overall motion that is performed.

After training, the best accuracy on the test set was 5.7471% with a validation accuracy of 6.8966% and a training accuracy of 45.6274%. This is a simple classification accuracy metric calculated by the number of correctly classified glosses divided by the total number of glosses tested. The validation accuracy was significantly higher in part 4, and this is because I wasn't including a final holdout set for testing. I selected 20% for the testing set because I wanted to be sure that at least some of the samples will be classified correctly, so a larger amount of data points can help confirm that.

The solution performs worse on the test set than it does on the validation set. This is likely because there is not a very large amount of training data. As a result, the model does not learn to generalize well, especially when some of that data is set aside for validation and testing. The model suffers from an extreme lack of data, so the number one fix for improving both training and validation performance is increasing the number of videos used. More data exists in this dataset, but it is hosted on YouTube, which blocks most download requests. In finding a workaround, it is possible to increase the number of videos per gloss by as much as 100-200%. An additional way to help improve testing accuracy is improving the current data augmentation practices, which currently just add noise to a few of the data samples. By implementing time dilation and possibly adding pretext tasks for self-supervised training, the model would be able to more appropriately generalize to generate accurate classifications. Since the data is randomly sampled, I think the testing and validation sets will likely have similar accuracy throughout training, especially since the validation data is not used to inform the gradient descent of the network in the training phase.

## [Short video](https://youtu.be/rS82PpKuIQE)

## Demo

To install the necessary files, use the `requirements.txt` file for installation:

```
python3 -m pip install -r requirements.txt
```

Then, just run the `demo.py` file as follows:

```
python3 demo.py
```

This file will take two data samples (with ID 56839 for gloss "tall" and ID 66639 for gloss "Thursday") and run them using the best model weights that I trained. The necessary landmark files are included in the `asl-data` folder. The gloss for "tall" is classified correctly, but the gloss for "Thursday" was misclassified as "cousin". Below is the training data for the model used in the demo:

![alstfig](https://github.com/user-attachments/assets/cb7e86e7-6006-4959-891f-5a32d7976645)

It is easy to see that the network is overfitting. The dark blue line represents the validation accuracy and the light blue line represents the training accuracy. These are both on the scale of 0 to 100%, as shown on the left side of the graph. The scale on the right side of the graph is for the training loss, whose units are not correlated with the accuracy percentages. The training loss is just illustrative of the increase in the model's ability to accurately classify the training data. The dotted line shows the average validation accuracy of the last 10 epochs. The network saved as a result of this graph is at the peak point, which is around 7%.


# Part 4: Second update

## Classifier justification

For the static hand gesture recognition, I used a random forest classifier. I selected this structure because it is easy to use, runs quickly on my laptop, and required little tuning to get working. A more complex solution was not required because the data points were sufficiently distinct from one another and could be easily distinguished with binary logic.

For the ASL recognition, I used MediaPipe to extract the landmarks (same as in the static hand gesture recognition). Time-dependent features were then extracted using an LSTM with an attention component. The classifier for this method is a single fully-connected layer that connects the attention context layer to the output. For most of my testing, I used 39 classes. I used this method because the relationship between the attention context vector and the final classification is too complex for a random forest.

## Classification accuracy

The following is the performance of the network on a set of 39 unique glosses (ASL "words"). Random accuracy is 2.56%.

    Network params:
            Input size: 98
            Num layers: 5
            Hidden size: 128
            Learning rate: 0.0001
            Dropout: 0.2
            Batch size: 64
            Num epochs: 250

![fig-23 57](https://github.com/user-attachments/assets/6f044da0-a482-45e4-baf2-d4d032b4c24a)

| Peak training accuracy | Peak testing accuracy |
| --- | --- |
| 53.7143% | 15.7931% |

## Training commentary

The current solution shows extreme overfitting. This is likely due to the very small amount of data available to the network -- the most common classes only have 13 instances. The 39 glosses that I selected for training all have 10 or more instances in the dataset; all other glosses appear 9 or fewer times. To remedy this, I implemented several adjustments:

1. **Dropout** - The LSTM has a dropout layer between LSTM layers. By randomly removing neurons, the network cannot rely too heavily on any set of neurons.
2. **Noise** - `GlossDataset.py` includes an optional `NoiseWrapper` class, which is a simple wrapper of a PyTorch `Dataset` object that adds noise to the set of MediaPipe landmarks for each frame whenever a data point is requested. It has a 50% chance of occurring and only affects data in the training set.
3. **Attention** - At the advice of Dr. Czajka, I implemented an attention vector to improve performance on the small amount of data. I found the performance to be relatively similar to the LSTM without attention.

However, the performance on the testing set still levels off well before a high accuracy is achieved.

1. **Dataset size** - One clear improvement would be simply increasing the size of the dataset. This could be done by filming myself doing the sign language and preprocessing that data. However, this would be time consuming.
2. **Preprocessing** - Preprocessing techniques could also be improved; many of the videos in the original dataset have different aspect ratios, so finding a way to normalize each video and their associated landmarks could increase testing accuracy. I would achieve this by taking each set of landmarks and adjusting them to fit in a square. This way, even if the glosses are signed by people of different sizes, the gestures will be expressed as a fraction of the square, so spatial information is retained while disregarding absolute size.
3. **Hyperparameter tuning** - I could write my own grid search hyperparemeter tuning method or try to use an AutoML Python library to adjust things like the number of LSTM layers, amount of noise, hidden layer size, batch size, etc. I think this would be very helpful for finding better performance relatively quickly before the final presentation, but I think the biggest gains are made in increasing the amount of data.
4. **Semisupervised pseudo labels** - It could help to pretrain the network with pseudo-labeled data. To achieve this, I could split each video into smaller segments and add noise to train the network to find higher level features that are robust to noise. This would allow me to create a large amount of data then fine-tune on the dataset itself.

## Demo setup

The demo uses the gloss for "accident". It shows one example of a correct classification and one incorrect classification. It loads in a CSV file `processed-videos-demo.csv`, which contains the two examples, and uses the weights from `weights/10pct39cl.pth`.

```
python3 -m pip install -r requirements.txt
python3 demo.py
```

The output will show the predicted class vs. the actual class. Due to a lack of foresight, I didn't keep track of which items were from the training/testing set (it is randomized every time) so most of the examples should return the correct classification.

# Part 3: First update

## Data samples

| Original | Processed |
| --- | --- |
| ![image](https://github.com/user-attachments/assets/89abb0ce-be44-4391-b1de-6009e895ea4f) | ![image](https://github.com/user-attachments/assets/946ed048-1791-43f9-910e-af9de86412b5) |
| ![image](https://github.com/user-attachments/assets/e62fd780-496b-461a-a433-d4d7715f1196) | ![image](https://github.com/user-attachments/assets/956210a4-4410-4a79-9718-e28c29580879) |

## Pre-processing and feature extraction
The data for this project is a set of short videos (~1-2 seconds long) that consist of a single ASL gloss (word). Each video is preprocessed according to `preprocess.py`, where each frame is input to MediaPipe. The script outputs a CSV file for each video with 98 features of pose and hand landmarks represented as a fraction of video width and height. 7 landmarks represent the upper body and 21 landmarks represent each hand, totaling 49 landmarks, each with 2 x/y positions, totaling 98 features.

This feature extraction method is ideal for this project because MediaPipe's API is sophisticated and produces reliable landmark outputs. The resulting neural network can be more widely applied since the input data is not dependent on factors such as resolution or height/width ratios. Additionally, it makes the problem solution less of a black box, where feature extraction occurs implicitly somewhere within a convolutional neural network.

> The data used for this solution is currently insufficient. I downloaded and trained my network on [this WLASL dataset](https://dxli94.github.io/WLASL/). The downloading process for this dataset utilizes a Python Python script that remotely downloads a set of thousands of videos included in a JSON file. However, half or more of the videos are hosted on YouTube, which does not support remote video downloading, so all of these videos are skipped. As a result, I have a dataset of hundreds of ASL glosses with only a few examples of each. I was aware of this shortcoming in my **Part 2** submission but I did not realize the extent to which it would ultimately impact performance.

## Setup

The code is now split into two projects: one is my preliminary solution, based just on single-frame hand landmarking. This solution is functional, accurate, and can be run on the webcam with the following:

```
python3 -m pip install -r requirements.txt
python3 recognition.py
```

### Static recognition demo
![static](https://github.com/user-attachments/assets/7616318a-3700-413f-a246-ae51a508facb)


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
