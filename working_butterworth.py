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
	for i in range(3):
		FFT_pad_blurred_image[:,:,i] = np.fft.fft2(pad_blurred_image[:,:,i])
		FFT_pad_blurred_image[:,:,i] = np.fft.fftshift(FFT_pad_blurred_image[:,:,i])
		FFT_kernel[:,:,i] = np.fft.fft2(pad_kernel[:,:,i])
		FFT_kernel[:,:,i] = np.fft.fftshift(FFT_kernel[:,:,i])

	
	FFT_recovered_image = np.zeros((2*rows, 2*cols, 3), np.complex128)
	recovered_image = np.zeros((2*rows, 2*cols, 3), np.float64)
	for i in range(3):
		FFT_recovered_image[:,:,i] = np.divide(FFT_pad_blurred_image[:,:,i], FFT_kernel[:,:,i])
		FFT_recovered_image[:,:,i] = np.multiply(FFT_recovered_image[:,:,i], H)
		FFT_recovered_image[:,:,i] = np.fft.ifftshift(FFT_recovered_image[:,:,i])
		recovered_image[:,:,i] = np.abs(np.fft.ifft2(FFT_recovered_image[:,:,i]))

	result = recovered_image[0:rows, 0:cols, :]
	print(result.dtype)
	
	result = cv2.normalize(result, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
	ground_truth = cv2.normalize(ground_truth, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
	
	SSIM_R = ssim(result[:,:,0], ground_truth[:,:,0])
	SSIM_G = ssim(result[:,:,1], ground_truth[:,:,1])
	SSIM_B = ssim(result[:,:,2], ground_truth[:,:,2])

	SSIM = (SSIM_R + SSIM_G + SSIM_B)/3.0
	SSIM_list = []
	SSIM_list.append(SSIM)
	print(SSIM)
	
	return result

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


if __name__ == '__main__':
	global ground_truth
	blurred_image = cv2.imread("Blurry1_1.jpg", 1)
	blurred_image = cv2.normalize(blurred_image, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)

	ground_truth = cv2.imread("GroundTruth1_1_1.jpg", 1)
	ground_truth = cv2.normalize(ground_truth, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)

	kernel = cv2.imread("blur_kern_1.png", 1)
	interactive_value(blurred_image, kernel)
	
	# H = butterworth_filter(blurred_image)
	# R_ch, G_ch, B_ch = cv2.split(blurred_image)

	# kernel = cv2.imread("cho.jpg", 1)
	# # kernel = cv2.resize(kernel, (21,21, ))
	# print("shape of kernel", kernel.shape)

	# kernel_R, kernel_G, kernel_B = cv2.split(kernel)

	# rec_R_ch = inverse_filter(R_ch, kernel_R, H)
	# rec_R_ch = rec_R_ch/np.max(rec_R_ch)

	# rec_G_ch = inverse_filter(G_ch, kernel_G, H)
	# rec_G_ch = rec_G_ch/np.max(rec_G_ch)
	
	# rec_B_ch = inverse_filter(B_ch, kernel_B, H)
	# rec_B_ch = rec_B_ch/np.max(rec_B_ch)

	# recovered_image = cv2.merge([rec_R_ch, rec_G_ch, rec_B_ch])
	# print("minimum value of recovered_image", np.min(recovered_image))
	# print("maximum value of recovered_image", np.max(recovered_image))

	# MSE = np.mean((GroundTruth - recovered_image)**2)
	# PSNR = 20*np.log10(np.max(GroundTruth)/np.sqrt(MSE))
	# print("the final PSNR value", PSNR)
	
	# cv2.imshow("recovered_image", recovered_image)
	# cv2.waitKey(0)