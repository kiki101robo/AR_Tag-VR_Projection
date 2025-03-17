import cv2
import numpy as np
from scipy import fftpack
from matplotlib import pyplot as plt

# FFT Class
class Background_FFT:
    def __init__(self, vid_path):
        self.video_feed = cv2.VideoCapture(vid_path)

    # Resize the frame
    @staticmethod
    def resize_frame(img, scale_percent=50):
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    # Separate background with fft
    def separate_background_w_fft(self, frame, filter_size=50):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        im_fft = fftpack.fft2(frame)

        # Creating a boolean high-pass filter
        rows, cols = frame.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.ones((rows, cols), np.uint8)
        mask[crow-filter_size:crow+filter_size, ccol-filter_size:ccol+filter_size] = 0

        # Applying the mask directly
        fshift = im_fft * mask

        # Inverse FFT
        ifft = np.abs(fftpack.ifft2(fshift))

        # Normalization and thresholding
        ifft = np.uint8(255 * (ifft - np.min(ifft)) / (np.max(ifft) - np.min(ifft)))
        _, mat = cv2.threshold(ifft, 150, 255, cv2.THRESH_BINARY)
        mat = cv2.dilate(mat, (57, 57))

        return mat


if __name__ == '__main__':
    from os import system
    system('cls')

    # Run FFT for a single frame
    fft = Background_FFT('Tag0.mp4')
    _, frame = fft.video_feed.read()
    img = fft.separate_background_w_fft(frame)
    cv2.imwrite("fft_background.png", img)

    cv2.imshow("frame", img)
    cv2.waitKey(0)



