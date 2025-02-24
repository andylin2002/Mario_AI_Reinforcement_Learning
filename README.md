## Preprocessing Frames for Deep Learning

This repository provides a simple function for preprocessing video frames, making them suitable for deep learning applications such as reinforcement learning and computer vision.

## Features
- Converts RGB images to grayscale
- Resizes images to 84x84 pixels
- Normalizes pixel values to the range [0,1]

## Installation
This project requires Python and OpenCV. Install the dependencies using:
```bash
pip install numpy torch opencv-python
```

## Usage
```python
import cv2
import numpy as np

def preprocess_frame(frame):
    """
    Preprocess a video frame by converting it to grayscale, resizing it,
    and normalizing the pixel values.
    
    Parameters:
        frame (numpy.ndarray): Input RGB frame.
    
    Returns:
        numpy.ndarray: Processed frame with shape (84, 84).
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    frame = cv2.resize(frame, (84, 84))  # Resize to 84x84 pixels
    frame = frame.astype(np.float32) / 255.0  # Normalize pixel values
    return frame

# Example usage
frame = cv2.imread("example.jpg")  # Load an example image
processed_frame = preprocess_frame(frame)
```

## Applications
This preprocessing function is useful for:
- Deep reinforcement learning (e.g., processing frames for an agent)
- Image-based neural networks that require grayscale input
- Efficiently reducing computational complexity in vision tasks

## Contributing
Feel free to submit pull requests or report issues.

## License
This project is licensed under the MIT License.

