import cv2
import numpy as np
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path", type=str, default="data/images_panaroma",
        help="path to images for panaroma construction")
    parser.add_argument(
        "--output_overlap", type=str, default="./overlap.txt",
        help="path to the overlap result")
    parser.add_argument(
        "--output_panaroma", type=str, default="./result.png",
        help="path to final panaroma image ")

    args = parser.parse_args()
    return args


def match(des1, des2):
    result = []
    for i, d in enumerate(des1):
        x = np.tile(d, (des2.shape[0], 1))
        dist = np.linalg.norm(x - des2, axis=1)
        k_idx = np.argsort(dist)[: 2]
        dist = dist[k_idx]
        temp = {'dist': dist, 'queryIdx': i, 'trainIdx': k_idx[0]}
        result.append(temp)
    return result


def stich_images(left, right):
    gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray_left, None)
    kp2, des2 = sift.detectAndCompute(gray_right, None)

    matches = match(des1, des2)

    good = []
    for m in matches:
        if m['dist'][0] < 0.8 * m['dist'][1]:
            good.append(m)
    matches = np.asarray(good)

    if len(matches) >= 4:
        src = np.float32([kp1[m['queryIdx']].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m['trainIdx']].pt for m in matches]).reshape(-1, 1, 2)
        H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    else:
        print('Not enough features to compute Homography')
        return

    r1, c1 = right.shape[:2]
    r2, c2 = left.shape[:2]

    p1 = np.array([[0, 0], [0, r1], [c1, r1], [c1, 0]]).reshape(-1, 1, 2).astype('float32')
    t_p = np.array([[0, 0], [0, r2], [c2, r2], [c2, 0]]).reshape(-1, 1, 2).astype('float32')
    p2 = cv2.perspectiveTransform(t_p, H)
    p = np.vstack((p1, p2))

    [r_min, c_min] = np.array(p.min(axis=0).ravel()).astype(int)
    [r_max, c_max] = np.array(p.max(axis=0).ravel()).astype(int)

    translation = [-r_min, -c_min]

    H_translation = np.array([[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]])

    dst = cv2.warpPerspective(left, H_translation.dot(H), (r_max - r_min, c_max - c_min))
    dst[translation[1]:r1 + translation[1], translation[0]:c1 + translation[0]] = right

    y_nonzero, x_nonzero, _ = np.nonzero(dst > 0)
    dst = dst[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]

    return dst

def stitch(inp_path, imgmark, N=4, savepath=''):
    # Load images in one go
    imgpath = [f'{inp_path}/{imgmark}_{n}.png' for n in range(1, N + 1)]
    imgs = [cv2.imread(ipath) for ipath in imgpath]

    # Compute grayscale and SIFT features simultaneously
    sift = cv2.SIFT_create()
    gray_imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]
    kps, desc = zip(*[sift.detectAndCompute(gray_img, None) for gray_img in gray_imgs])

    # Calculate overlaps
    overlap_arr = np.identity(len(imgs))
    pairs = []

    for i in range(N - 1):
        for j in range(i + 1, N):
            matches = [m for m in match(desc[i], desc[j]) if m['dist'][0] < 0.8 * m['dist'][1]]

            x = len(matches) / len(kps[i])
            y = len(matches) / len(kps[j])

            if 0.1 < x < 0.9 and 0.1 < y < 0.9:
                overlap_arr[i][j] = overlap_arr[j][i] = 1
                pairs.append((i, j))

    # Determine the stitching order
    order = sorted(pairs, key=lambda x: -len(match(desc[x[0]], desc[x[1]])))

    # Stitch images based on the order
    result = imgs[order[0][0]]
    for o in order:
        result = stich_images(result, imgs[o[1]])

    cv2.imwrite(savepath, result)

    return overlap_arr

if __name__ == "__main__":
    args = parse_args()
    overlap_arr = stitch(args.input_path, 't2', N=4, savepath=args.output_panaroma)
    with open(args.output_overlap, 'w') as outfile:
        json.dump(overlap_arr.tolist(), outfile)
