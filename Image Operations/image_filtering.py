import cv2
import numpy as np

def convolve(image, kernel):
    image = image.astype('float32') / 255.0
    k_size = kernel.shape[0]
    padding_width = (k_size - 1) // 2
    temp_img = np.pad(image, ((padding_width, padding_width), (padding_width, padding_width)), mode='constant')
    
    h, w = image.shape
    convolved_img = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            convolved_img[i, j] = np.sum(temp_img[i:i + k_size, j:j + k_size] * kernel)

    return (convolved_img * 255).clip(0, 255)

def correlate(image, kernel):
    return convolve(image, np.flip(kernel))

def median_filter5x5(image):
    h, w = image.shape
    padded_image = np.pad(image, ((2, 2), (2, 2)), mode='constant')
    median_img = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            median_img[i, j] = np.median(padded_image[i:i + 5, j:j + 5])

    return median_img.astype(np.uint8)

def gamma(image, g):
    image = image.astype('float32') / 255.0
    out_img = np.power(image, g)
    return (out_img * 255).astype(np.uint8)

if __name__ == '__main__':
    img = cv2.imread('Noisy_image.png', 0)
    
    # Convolution filter
    conv_filter = np.ones((3,3)) / 9
    conv_img = convolve(img, conv_filter)
    cv2.imwrite('convolved_image.png', conv_img)

    # Averaging filter (which is the same as the convolution filter in this case)
    avg_img = correlate(img, conv_filter)
    cv2.imwrite('average_image.png', avg_img)

    # Gaussian Filter
    g_filter = np.array([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 16
    g_img = correlate(img, g_filter)
    cv2.imwrite('gaussian_image.png', g_img)

    # Median filter 5x5
    m_img = median_filter5x5(img)
    cv2.imwrite('median_image.png', m_img)

    img = cv2.imread('Uexposed.png', 0)
    gamma_img = gamma(img, 0.4)
    cv2.imwrite('adjusted_image.png', gamma_img)
