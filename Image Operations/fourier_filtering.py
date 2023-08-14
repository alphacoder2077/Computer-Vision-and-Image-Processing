import cv2
import numpy as np

def fourier_transform(image, s):
    h, w = image.shape
    image = image.astype('float32')
    
    # Padding
    pad_h, pad_w = h, w  # Adjusted for clarity; P x Q becomes 2h x 2w
    padded_img = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), 'constant', constant_values=0)
    
    # DFT
    dft = cv2.dft(padded_img, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    # Create a meshgrid
    x = np.linspace(-pad_w, pad_w, 2*pad_w)
    y = np.linspace(-pad_h, pad_h, 2*pad_h)
    x, y = np.meshgrid(x, y)
    distance = np.sqrt(x**2 + y**2)
    
    # Gaussian filter
    g_filter = np.exp(-distance**2 / (2 * s**2))
    g_filter = np.repeat(g_filter[:, :, np.newaxis], 2, axis=2)  # Broadcast along the third dimension
    
    # Apply filter and inverse DFT
    filtered_dft_shift = dft_shift * g_filter
    inverse_dft_shift = np.fft.ifftshift(filtered_dft_shift)
    inverse_img = cv2.idft(inverse_dft_shift, flags=cv2.DFT_REAL_OUTPUT)
    
    # Crop image back to original size and normalize
    cropped_img = inverse_img[pad_h:-pad_h, pad_w:-pad_w]
    cropped_img = 255 * cropped_img / np.max(cropped_img)
    
    # Log magnitude spectrum
    magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    magnitude_spectrum = np.log(1 + magnitude) 
    magnitude_spectrum = 255 * magnitude_spectrum / np.max(magnitude_spectrum)
    
    return cropped_img, magnitude_spectrum

if __name__ == '__main__':
    img = cv2.imread('Noisy_image.png', 0)

    image, dft = fourier_transform(img, s=220)
    cv2.imwrite('converted_fourier.png', dft)
    cv2.imwrite('gaussian_fourier.png', image)

