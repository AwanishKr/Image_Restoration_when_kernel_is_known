import numpy as np
import cv2
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from skimage.measure import structural_similarity as ssim

def inverse_filter(blurred_image, kernel, value):
	global ground_truth
	rows, cols, channels = blurred_image.shape
	H = np.zeros((2*rows, 2*cols), np.float64)
	
	print("value of radius",value)
	value_list = []
	value_list.append(value)
	# implementing the mathematical formula for a lowpass butterworth filter given in book digital image processing by gonzalese
	for u in range(0, 2*rows):
		for v in range(0, 2*cols):
			D = np.sqrt((u-rows)**2 + (v-cols)**2)
			H[u,v] = 1/(1+(D/value)**20)

	pad_blurred_image = np.zeros((2*rows, 2*cols, 3), np.float64)
	pad_blurred_image[0:blurred_image.shape[0], 0:blurred_image.shape[1], :] = blurred_image

	# padding of kernel
	pad_kernel = np.zeros((2*rows, 2*cols, 3), np.float64)
	pad_kernel[0:kernel.shape[0], 0:kernel.shape[1], :] = kernel

	FFT_pad_blurred_image = np.zeros((2*rows, 2*cols, 3), np.complex128)
	FFT_kernel = np.zeros((2*rows, 2*cols, 3), np.complex128)
	# taking fft channelwise and shifting the spectrum to center of frequency domain
	for i in range(3):
		FFT_pad_blurred_image[:,:,i] = np.fft.fft2(pad_blurred_image[:,:,i])
		FFT_pad_blurred_image[:,:,i] = np.fft.fftshift(FFT_pad_blurred_image[:,:,i])
		FFT_kernel[:,:,i] = np.fft.fft2(pad_kernel[:,:,i])
		FFT_kernel[:,:,i] = np.fft.fftshift(FFT_kernel[:,:,i])

	
	FFT_recovered_image = np.zeros((2*rows, 2*cols, 3), np.complex128)
	recovered_image = np.zeros((2*rows, 2*cols, 3), np.float64)
	# division then lowpass filtering and then inverse shift and then inverse fft
	for i in range(3):
		FFT_recovered_image[:,:,i] = np.divide(FFT_pad_blurred_image[:,:,i], FFT_kernel[:,:,i])
		FFT_recovered_image[:,:,i] = np.multiply(FFT_recovered_image[:,:,i], H)
		FFT_recovered_image[:,:,i] = np.fft.ifftshift(FFT_recovered_image[:,:,i])
		recovered_image[:,:,i] = np.abs(np.fft.ifft2(FFT_recovered_image[:,:,i]))

	result = recovered_image[0:rows, 0:cols, :]
	print(result.dtype)
	
	# since the ssim is calculated over normalized image
	result = cv2.normalize(result, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
	ground_truth = cv2.normalize(ground_truth, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
	
	#my ssim wasn't working on coloured images so i splitted them and taken average
	SSIM_R = ssim(result[:,:,0], ground_truth[:,:,0])
	SSIM_G = ssim(result[:,:,1], ground_truth[:,:,1])
	SSIM_B = ssim(result[:,:,2], ground_truth[:,:,2])

	SSIM = (SSIM_R + SSIM_G + SSIM_B)/3.0
	# i printed values of ssim and values of parameters to compare when does it gives best ssim
	print(SSIM)
	
	return result

# I took this function from internet and modified it
def interactive_value(blurred_image, kernel):
    D_min = 5
    D_max = 1600
    D_init = 10
    fig = plt.figure()
    plt.axis("off")
    recovered_image = inverse_filter(blurred_image,kernel, D_init)
    recovered_image = cv2.normalize(recovered_image, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
    # recovered_image_plot = plt.imshow(recovered_image)
    recovered_image_plot = plt.imshow(cv2.cvtColor(recovered_image.astype(np.float32), cv2.COLOR_BGR2RGB))
    slider_ax = plt.axes([0.1, 0.05, 0.8, 0.05])
    value_slider = Slider(slider_ax, 'value', D_min, D_max, valinit=D_init)
     # D0_slider.on_changed(update)
    def update(value):
        recovered_image = inverse_filter(blurred_image, kernel, value).astype(np.float32)
        recovered_image = cv2.cvtColor(recovered_image, cv2.COLOR_BGR2RGB)
        recovered_image = cv2.normalize(recovered_image, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
        recovered_image_plot.set_data(recovered_image)
        fig.canvas.draw_idle()
    value_slider.on_changed(update)
    plt.show()

# this function was written by me
if __name__ == '__main__':
	# global because i am accessing it in another function
	global ground_truth
	blurred_image = cv2.imread("Blurry1_1.jpg", 1)
	blurred_image = cv2.normalize(blurred_image, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)

	ground_truth = cv2.imread("GroundTruth1_1_1.jpg", 1)
	ground_truth = cv2.normalize(ground_truth, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)

	kernel = cv2.imread("blur_kern_1.png", 1)
	interactive_value(blurred_image, kernel)
