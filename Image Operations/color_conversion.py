import cv2
import numpy as np
import math

def rgb_to_hsv(image):
    R = image[:,:,2]
    G = image[:,:,1]
    B = image[:,:,0]
    
    V = np.max(image, axis=2)
    m = np.min(image, axis=2)
    diff = V - m

    S = np.where(V != 0, diff / V, 0)
    
    H = np.zeros_like(V)

    condition = V == R
    H[condition] = (60 * (G[condition] - B[condition]) / diff[condition]) % 360
    
    condition = V == G
    H[condition] = (120 + 60 * (B[condition] - R[condition]) / diff[condition]) % 360
    
    condition = V == B
    H[condition] = (240 + 60 * (R[condition] - G[condition]) / diff[condition]) % 360

    H = H/2
    return np.stack([H, S*255, V], axis=-1)

def rgb_to_lab(image):
    def f(t):
        return np.where(t > 0.008856, np.power(t, 1/3.0), 7.787*t + 16/116)

    R = image[:,:,2]
    G = image[:,:,1]
    B = image[:,:,0]

    mat = np.array([[0.412453, 0.357580, 0.180423],
                    [0.212671, 0.715160, 0.072169],
                    [0.019334, 0.119193, 0.950227]])
    
    XYZ = np.dot(image.reshape((-1, 3)), mat.T).reshape(image.shape)
    
    X, Y, Z = XYZ[:,:,0], XYZ[:,:,1], XYZ[:,:,2]
    
    X /= 0.950456
    Z /= 1.088754

    L = np.where(Y > 0.008856, 116 * np.power(Y, 1/3) - 16, 903.3 * Y)
    a = 500 * (f(X) - f(Y))
    b = 200 * (f(Y) - f(Z))

    return np.stack([L*255/100, a+128, b+128], axis=-1)

def rgb_to_cmyk(image):
    R = image[:,:,2]
    G = image[:,:,1]
    B = image[:,:,0]
    
    K = 1 - np.max(image, axis=2)
    C = np.where(K != 1, (1 - R - K) / (1 - K), 0)
    M = np.where(K != 1, (1 - G - K) / (1 - K), 0)
    Y = np.where(K != 1, (1 - B - K) / (1 - K), 0)
    
    return np.stack([C*255, M*255, Y*255, K*255], axis=-1)

if __name__ == '__main__':
    img = cv2.imread('Lenna.png').astype('float32') / 255.0

    HSV_img = rgb_to_hsv(img)
    Lab_img = rgb_to_lab(img)
    CMYK_img = rgb_to_cmyk(img)

    cv2.imwrite("hsv_image.png", HSV_img)
    cv2.imwrite("lab_image.png", Lab_img)
    cv2.imwrite("cmyk_image.png", CMYK_img)
