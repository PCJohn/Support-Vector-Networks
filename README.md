# Support-Vector-Networks

This is a module to detect humans using SVMs. The module may be used to train a binary classifier to human/non-human classification tasks. It can also be applied to run human tracking on a video output from a stationary camera.

**Requirements**

    Scikit-Learn
    OpenCV for Python

Note that you must have FFmpeg installed to load the video files in OpenCV. You can accomplish this with

	brew install opencv3 --with-ffmpeg

Given the differences in OpenCV versions 2 and 3, we provide equivalent scripts for both versions.

**Dataset**

We use the INRIA person dataset: http://pascal.inrialpes.fr/data/human/. The module has a pretrained model: human_model.pkl, which was trained on 2000 images from this dataset (1000 positive and 1000 negative).

Following the procedure in the paper "Histograms of Oriented Gradients for Human Detection" (https://courses.engr.illinois.edu/ece420/fa2017/hog_for_human_detection.pdf), the input images were resized to size (64 x 128) before feature extraction. We use histogram-of-gradient features (provided in the cv2 library) to convert the input into vectors of dimension 3780.
The negative image consisted of negative samples provided in the dataset. Along with this, we use small, randomly cropped subimages (boxes of size (8 x 16) and (32 x 64)) from these negative samples.

**Training**

The model was trained using the Scikit-Learn SVM library. Running a grid search, we found the ideal parameters to be C=5.0 and Gamma=0.01. This gives a training accuracy of 100 and a validation accurcay of 97.75%. To train a new model using the module, set the path to save the new model in the variable 'model' in human.py (model = '<path_to_model>') and run:

    python human.py train

This will load the dataset, train the SVM and save the model to the specified location.

**Video processing**

We use the background subtractor provided in the cv2 library and experimented with background SubtractorMOG and backgroundSubtractorKNN (available in OpenCV version 3). These output frames with a binary mask indicating the position of moving objects (foreground). Morphological dilation with blurring followed by thresholding was used to connected the generated contours. The parameters for these operations are application specific and may need to be tuned depending on the type of video used an input.

We ran the module on a set of videos. The output can be seen here: <link to youtube>
