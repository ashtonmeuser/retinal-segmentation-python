# Retinal Blood Vessel Segmentation Using Line Operators and Support Vector Classification

## Getting Started

Before images can be segmented, dependencies must be installed and a virtual environment should be created.

### Prerequisites

Python 3.4 or greater and PIP must be installed. For installation instructions, refer to [official documentation](https://www.python.org/downloads/).

Clone the repository.

A virtual environment is advised when installing dependencies. In the terminal, navigate to the project directory. Run `python3 -m venv venv`. To activate the virtual environment, run `source venv/bin/activate`.

### Dependencies

Dependencies are found in *requirements.txt*. Run `pip install -r requirements.txt` to install.

## Running the Program

The program provides two primary functions: training the model and segmenting image pixels. A model must be trained using the former in order to be used in the latter.

### Arguments

All arguments apply to training and classification besides the `-t` argument, which determines whether the program is to train the model or use the trained model to segment an image.

The same settings should be used for kernel size and rotational resolution when training the model and segmenting an image.

| Symbol | Name     | Description
|--------|----------|-------
| **i**  | images   | Images to be used in training the model or an image to be classified
| **k**  | kernel   | The neighborhood size used in line detection
| **r**  | rotation | Rotational resolution used in line detection
| **t**  | train    | Instruct program to train the model
| **s**  | save     | Save the segmented image as a PNG
| **d**  | display  | Display the segmented image
| **v**  | verbose  | Print all logging messages instead of only high-level information

### Training the Model

Before the images can be segmented, the model must be trained. This is the most time-consuming portion of running the program. To save time, several example models are included, pretrained, in the repository. These examples can be renamed *model.p* in order to be used in segmentation of fundus images.

To train the model, run `main.py -i X -t` where *X* is one or more image numbers. The image numbers represent the suffix of the image in the [DRIVE database](https://www.isi.uu.nl/Research/Databases/DRIVE/), and is therefore limited to integers 0 through 40. Note that images 21 through 40 are typically used as a training set.

Training the model with a full training set of 20 images can take quite a while. The writer's machine (2nd gen. i5, 16GB RAM) required nearly 19 hours to train a binary SVM using images 21 through 40. The probabilistic model used in this repo would likely take at least twice that time on the same machine. Trainng a model using one or two images should take no more than 1.5 hours.

### Segmenting Fundus Images

Only one image at a time can be supplied to be segmented. Run `main.py -i X -t` where *X* is exactly one image number. The image numbers represent the suffix of the image in the [DRIVE database](https://www.isi.uu.nl/Research/Databases/DRIVE/), and is therefore limited to integers 0 through 40. Note that images 0 through 20 are typically used as a testing set.

Set the `-s` and/or `-d` arguments to save and/or display the segmented image after classification.
