# Image Processing Functions

This repository contains custom Python functions for various image processing tasks. Each function accepts an image as a numpy array and performs a specific operation.

### 1. Grayscale conversion

`gray(img, method='luminosity')`

This function converts a colored image into grayscale. It supports three methods: 'average', 'lightness', and 'luminosity'. The default method is 'luminosity'.

**Example:**

```
gray_image = gray(image, method='lightness')
```

![Output](results/gray_image.png) 


### 2. Translation

`translation(img, tr_factor)`

This function translates an image by a specified factor along both axes.

**Example:**

```
translated_image = translation(image, tr_factor=50)
```

![Output](results/translate_image.png) 


### 3. Vertical Flip

`flip_vertical(img)`

This function flips an image vertically.

**Example:**

```
flipped_image = flip_vertical(image)
```

![Output](results/vertical_image.png) 


### 4. Horizontal Flip

`flip_horizontal(img)`

This function flips an image horizontally.

**Example:**

```
flipped_image = flip_horizontal(image)
```

![Output](results/horizontal_image.png)


### 6. Scaling

`scaling(img, new_w, new_h, method='bilinear')`

This function scales an image to a new size using either bilinear interpolation or nearest neighbor interpolation. 

**Bilinear Interpolation** is a resampling technique that utilizes the weighted average of the four nearest pixel values to compute the output pixel value. It provides smoother results compared to the nearest neighbor interpolation.

**Nearest Neighbor Interpolation** is a simple method where the value of an output pixel is assigned the value of the pixel that is nearest to the corresponding input pixel. It's less compute-intensive but may produce blocky results.

**Example:**

```
scaled_image = scaling(image, new_w=200, new_h=200, method='bilinear')
```

![Output](results/scale_image.png)


### 7. Rotation

`rotate(img, angle, method='bilinear')`

This function rotates an image by a specified angle using either bilinear interpolation or nearest neighbor interpolation.

**Example:**

```
rotated_image = rotate(image, angle=45, method='bilinear')
```

![Output](results/rotate_image.png)

---
