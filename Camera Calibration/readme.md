## Camera Calibration using Chessboard Patterns

This repository contains code for camera calibration tasks using chessboard patterns. The calibration process helps to estimate the parameters of the camera, which can be used for various computer vision tasks like 3D reconstruction, structure from motion, and more.

### Structure

The repository contains two main tasks:

1. **Rotation Matrix Computation** (`task1.py`)
2. **Chessboard Corner Detection & Parameter Estimation** (`task2.py`)

---

## Task 1: Rotation Matrix Computation

### Overview

This task deals with computing the rotation matrices. Given the rotation angles along the x, y, and z axes, the script provides functionality to compute the rotation matrix from one coordinate system to another and vice-versa.

### Functions

1. `findRot_xyz2XYZ(alpha, beta, gamma) -> np.ndarray`:
    - **Input**: Rotation angles (`alpha`, `beta`, `gamma`) along x, y, and z axes respectively (in degrees).
    - **Output**: 3x3 numpy array representing the rotation matrix from `xyz` to `XYZ` coordinates.

2. `findRot_XYZ2xyz(alpha, beta, gamma) -> np.ndarray`:
    - **Input**: Rotation angles (`alpha`, `beta`, `gamma`) for the 3 steps respectively (in degrees).
    - **Output**: 3x3 numpy array representing the rotation matrix from `XYZ` to `xyz` coordinates.

---

## Task 2: Chessboard Corner Detection & Parameter Estimation

### Overview

This task is centered around detecting corners in an image of a chessboard pattern and using these corners to compute camera parameters. This is a fundamental step in camera calibration.

### Functions

1. `find_corner_img_coord(image) -> np.ndarray`:
    - **Input**: Image (`image`) of size MxNx3.
    - **Output**: numpy array of size 32x2 representing the pixel coordinates of the 32 detected chessboard corners.

2. `find_corner_world_coord(img_coord) -> np.ndarray`:
    - **Input**: Image coordinates (`img_coord`) of the corners.
    - **Output**: numpy array of size 32x3 representing the world coordinates of the 32 chessboard corners.

3. `find_intrinsic(img_coord, world_coord) -> Tuple[float, float, float, float]`:
    - **Input**: Image coordinates (`img_coord`) of the 32 corners and world coordinates (`world_coord`) of the 32 corners.
    - **Output**: Focal lengths (`fx`, `fy`) and principal point of the camera (`cx`, `cy`).

4. `find_extrinsic(img_coord, world_coord) -> Tuple[np.ndarray, np.ndarray]`:
    - **Input**: Image coordinates (`img_coord`) of the 32 corners and world coordinates (`world_coord`) of the 32 corners.
    - **Output**: Rotation matrix (`R`) and translation matrix (`T`) representing the extrinsic camera parameters.

---

## Usage

Ensure that all necessary libraries such as `numpy` and `opencv` are installed.

Run the corresponding tasks using:

```bash
python task1.py
```

or 

```bash
python task2.py
```

Provide appropriate inputs as required.

---
