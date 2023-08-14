# Image Stitching for Panorama Construction

This repository contains a Python implementation of image stitching, which is an essential step in constructing panoramic images. Given a set of overlapping images, the code computes feature correspondences using the SIFT algorithm and determines the stitching order based on these correspondences. The images are then merged together to produce a seamless panorama.

## Prerequisites

Ensure you have the following libraries installed:
- OpenCV (cv2)
- NumPy (numpy)
- argparse
- json

You can install them using pip:
```bash
pip install opencv-python numpy argparse
```

## How to Use

1. **Run the Code**
To stitch images together, run:
```bash
python image_stitching.py --input_path path_to_images_directory
```
By default, the program expects images to be named as `t2_1.png`, `t2_2.png`, etc. You can change this by modifying the `imgmark` argument in the code.

3. **Optional Arguments**
- `--output_overlap`: Specify the path to save the overlap relation in a text file. Defaults to `./task2_overlap.txt`.
- `--output_panaroma`: Specify the path to save the final panorama image. Defaults to `./task2_result.png`.

For example:
```bash
python image_stitching.py --input_path ./data/images_panaroma --output_overlap ./overlap.txt --output_panaroma ./panorama.png
```

## Functionality

- `parse_args()`: Parses command line arguments.
- `match(des1, des2)`: Computes the correspondences between two sets of SIFT descriptors.
- `stich_images(left, right)`: Stitches two images together based on their SIFT features.
- `stitch(inp_path, imgmark, N=4, savepath='')`: The main function that computes the stitching order of images and merges them to produce the final panorama.