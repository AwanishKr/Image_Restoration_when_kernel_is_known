import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def constrained_LeastSquare(blurred_channel, kernel, value):
	rows, cols, channel = blurred_channel.shape
	pad_blurred_channel = np.zeros((2*rows, 2*cols, 3), np.float64)
	pad_blurred_channel[0:blurred_channel.shape[0], 0:blurred_channel.shape[1], :] = blurred_channel
	
	FFT_pad_blurred_channel = np.zeros((2*rows, 2*cols, 3), np.complex64)
	
	for i in range(3):
		FFT_pad_blurred_channel[:,:,i] = np.fft.fftshift(np.fft.fft2(pad_blurred_channel[:,:,i]))

	pad_kernel = np.zeros((2*rows, 2*cols, 3), np.float64)
	pad_kernel[0:kernel.shape[0], 0:kernel.shape[1], :] = kernel
	
	FFT_kernel = np.zeros((2*rows, 2*cols, 3), np.complex64)
	for i in range(3):
		FFT_kernel[:,:,i] = np.fft.fftshift(np.fft.fft2(pad_kernel[:,:,i]))

	P_xy = np.array([[0,-1, 0],
					 [-1,4,-1],
					 [0,-1, 0]], np.float64)
	
	padded_P_xy = np.zeros((2*rows, 2*cols), np.float64)
	padded_P_xy[0:3, 0:3] = P_xy
	P_uv = np.fft.fftshift(np.fft.fft2(padded_P_xy))
	
	constrained_filter = np.zeros((2*rows, 2*cols, 3), np.complex64)
	# gamma = np.average((np.abs(FFT_kernel))**2)
	# optimal gamma = 7,38,508
	gamma = value
	for i in range(3):
		constrained_filter[:,:,i] = np.divide(np.conjugate(FFT_kernel[:,:,i]), ((np.abs(FFT_kernel[:,:,i]))**2 + gamma*(np.abs(P_uv))**2))

	FFT_recovered_channel = np.zeros((2*rows, 2*cols, 3), np.complex64)
	recovered_channel = np.zeros((2*rows, 2*cols, 3), np.float64)
	for i in range(3):
		FFT_recovered_channel[:,:,i] = np.multiply(FFT_pad_blurred_channel[:,:,i], constrained_filter[:,:,i])
		recovered_channel[:,:,i] = np.abs(np.fft.ifft2(np.fft.ifftshift(FFT_recovered_channel[:,:,i])))
	
	result = recovered_channel[0:rows, 0:cols, :]
	
	# result = cv2.normalize(result, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
	# ground_truth = cv2.normalize(ground_truth, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
	
	# result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
	# ground_truth_gray = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY)
	# SSIM = ssim(result_gray, ground_truth_gray)
	# print(SSIM)

	return result

def interactive_value(blurred_image, kernel):
    D_min = 10000
    D_max = 758508
    D_init = 820000
    fig = plt.figure()
    plt.axis("off")
    recovered_image = constrained_LeastSquare(blurred_image,kernel, D_init)
    recovered_image = cv2.normalize(recovered_image, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
    # recovered_image_plot = plt.imshow(recovered_image)
    recovered_image_plot = plt.imshow(cv2.cvtColor(recovered_image.astype(np.float32), cv2.COLOR_BGR2RGB))
    slider_ax = plt.axes([0.1, 0.05, 0.8, 0.05])
    value_slider = Slider(slider_ax, 'value', D_min, D_max, valinit=D_init)
     # D0_slider.on_changed(update)
    def update(value):
        recovered_image = constrained_LeastSquare(blurred_image, kernel, value).astype(np.float32)
        recovered_image = cv2.cvtColor(recovered_image, cv2.COLOR_BGR2RGB)
        recovered_image = cv2.normalize(recovered_image, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
        recovered_image_plot.set_data(recovered_image)
        fig.canvas.draw_idle()
    value_slider.on_changed(update)
    plt.show()


if __name__ == '__main__':
	blurred_image = cv2.imread("blurred_image.jpg", 1)
	blurred_image = cv2.normalize(blurred_image, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
	
	# ground_truth = cv2.imread("last_blur_1.png", 1)
	# ground_truth = cv2.normalize(ground_truth, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)

	kernel = cv2.imread("my_kernel.jpg", 1)
	kernel = cv2.resize(kernel, (21,21,))
	
	interactive_value(blurred_image, kernel)	

	# recovered_image = constrained_LeastSquare(blurred_image, kernel)
	# recovered_image = cv2.normalize(recovered_image, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
	
	# qrecovered_image = cv2.cvtColor(recovered_image, cv2.COLOR_BGR2RGB)
	# fig = plt.figure()
	# plt.axis("off")
	# plt.imshow(recovered_image)
	# plt.show()
