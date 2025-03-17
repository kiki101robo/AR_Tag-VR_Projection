# AR_Tag-VR_Projection

## Overview
The "AR_Tag-VR_Projection" project is designed to process AR tags within video streams to perform advanced visual augmentations. Using Python and OpenCV, this system processes video inputs to isolate dynamic features via FFT, overlays 2D images, and then projects 3D graphics onto detected AR tags.

## Features
- **FFT Background Isolation**: Applies Fast Fourier Transform techniques to separate moving elements from static backgrounds in video frames.
- **2D Overlay**: Enhances detected AR tags with a 2D image of Testudo, aligning it perfectly with the real-world orientation of the tags.
- **3D Projection**: Implements a VR cube projection on AR tags, showcasing potential in augmented reality applications through 3D modeling.

## Files
- `fft_background.py`: Handles the background isolation using FFT.
- `cornerfinder.py`: Detects corners which are crucial for defining the boundaries of AR tags.
- `read_ar.py`: Reads and decodes the AR tags from processed frames.
- `overlay_2D.py`: Manages the overlaying of 2D images onto detected tags.
- `overlay_3D.py`: Projects 3D cubes onto the AR tags based on their spatial orientation.

## Usage
To use this project, ensure you have the necessary dependencies installed, including Python, OpenCV, NumPy, SciPy, and scikit-learn. Run each script according to the needs of the particular stage of processing:

1. **FFT Background Isolation**: python fft_background.py
2. **Corner Detection**: cornerfinder.py
3. **2D Overlay on Tags**: python overlay_2D.py
4. **3D Cube Overlay on Tags**: python overlay_3D.py


### Results
![Input](/"Tag0-ezgif.com-video-to-gif-converter.gif")
![Testudo_Overlay](/"Tag0_testudo_overlay-ezgif.com-video-to-gif-converter.avi")




## Requirements
- Python 3.10
- OpenCV
- NumPy
- SciPy
- matplotlib
- scikit-learn

## Installation
Install the required packages using pip:
