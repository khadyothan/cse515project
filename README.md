# Multimedia Feature Extraction and Image Retrieval

This repository contains code snippets and functions for multimedia feature extraction and image retrieval tasks. It includes functions for feature extraction using color moments, HOG (Histogram of Oriented Gradients), and ResNet-50 pre-trained on ImageNet. It also includes a task to find the top K most similar images in a database given an input image.

## Code Structure

The code is organized into different modules, each serving a specific purpose:

### `color_moments.py`

This module contains a function `color_moments(image_id, image)` that calculates color moments feature descriptors for an input image.

### `HOG.py`

This module contains a function `HOG(image_id, image)` for computing Histogram of Oriented Gradients (HOG) feature descriptors for an input image.

### `resnet_features.py`

This module provides a function `resnet_features(image_id, image)` to extract feature descriptors from the ResNet-50 model pre-trained on ImageNet.

### `feature_descriptors_extraction.py`

This module contains a function `feature_descriptors_extraction(collection, dataset)` to extract and store feature descriptors for all images in a database.

### `task3.py`

This module includes a function `task3(input_image_id, k, collection, dataset)` to find the top K most similar images in the database to an input image using various feature descriptors.

### `print_top_k_images.py`

This module provides a function `print_images(image_dict, heading, target_size)` to display a grid of images along with their distance scores.

### `task1_printing.py`

This module includes functions `print_cm_grid`, `print_hog_grid`, and `readable_output` for printing color moments and HOG feature grids in a readable format.

## Usage

To use these functions and modules, you can follow these steps:

1. Import the required modules in your Python script or Jupyter Notebook.

2. Use the provided functions to perform specific tasks, such as feature extraction or image retrieval.

3. Customize the input and parameters as needed for your specific use case.

4. Run your script to perform the desired multimedia tasks.

## Example Usage

You can refer to the provided main program `main.py` for an example of how to use these functions to perform multimedia tasks, including feature extraction and image retrieval.

## Dependencies

The code relies on the following Python libraries:
- `numpy`
- `torch` (PyTorch)
- `torchvision`
- `PIL` (Python Imaging Library)
- `matplotlib`
- `pymongo` (for database operations, if used)

Ensure that you have these libraries installed in your Python environment.

## License

Feel free to customize and use it for your multimedia and image retrieval projects!
