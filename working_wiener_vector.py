import numpy as np
import cv2
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from skimage.measure import structural_similarity as ssim

# def ssim(im1, im2, k=(0.01, 0.03), l=255):
#     c1 = (k[0]*l)**2
#     c2 = (k[1]*l)**2
#     mu_im1 = np.mean(im1)
#     mu_im2 = np.mean(im2)
#     var_im1 = np.var(im1)
#     # var_im1 =(im1-mu_im1)**2/im1.size[0]**2
#     var_im2 = np.var(im2)
#     # var_im2 =(im2-mu_im2)**2/im2.size[0]**2
#     cov_im1_im2 = np.mean((im1-mu_im1)*(im2-mu_im2))

#     SSIM = ((2*mu_im1*mu_im2 + c1)*(2*cov_im1_im2 + c2))/((mu_im1**2 + mu_im2**2 + c1)*(var_im1 + var_im2 +c2))

#     return SSIM

def wiener_filter(blurred_image, kernel, value):
	global ground_truth
	rows, cols, channel = blurred_image.shape
	pad_blurred_image = np.zeros((2*rows, 2*cols, 3), np.float64)
	pad_blurred_image[0:blurred_image.shape[0], 0:blurred_image.shape[1], :] = blurred_image

	pad_kernel = np.zeros((2*rows, 2*cols, 3), np.float64)
	pad_kernel[0:kernel.shape[0], 0:kernel.shape[1], :] = kernel
	
	FFT_pad_blurred_image = np.zeros((2*rows, 2*cols, 3), np.complex128)
	FFT_kernel = np.zeros((2*rows, 2*cols, 3), np.complex128)
	for i in range(3):
		FFT_pad_blurred_image[:,:,i] = np.fft.fftshift(np.fft.fft2(pad_blurred_image[:,:,i]))
		FFT_kernel[:,:,i] = np.fft.fftshift(np.fft.fft2(pad_kernel[:,:,i]))

	# K = np.average((np.abs(FFT_kernel))**2)
	K = value
	print(K)
	# print(np.average((np.abs(FFT_kernel))**2))
	wiener = np.zeros((2*rows, 2*cols, 3), np.complex128)
	for i in range(3):
		wiener[:,:,i] = np.divide((np.abs(FFT_kernel[:,:,i]))**2, ((np.abs(FFT_kernel[:,:,i]))**2 + K))
		wiener[:,:,i] = np.divide(wiener[:,:,i], FFT_kernel[:,:,i])

	FFT_recovered_channel = np.zeros((2*rows, 2*cols, 3), np.complex128)
	recovered_channel = np.zeros((2*rows, 2*cols, 3), np.float64)
	for i in range(3):
		FFT_recovered_channel[:,:,i] = np.multiply(FFT_pad_blurred_image[:,:,i], wiener[:,:,i])
		FFT_recovered_channel[:,:,i] = np.fft.ifftshift(FFT_recovered_channel[:,:,i])
		recovered_channel[:,:,i] = np.abs(np.fft.ifft2(FFT_recovered_channel[:,:,i]))
	
	result = recovered_channel[0:rows, 0:cols, :]
	
	result = cv2.normalize(result, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
	ground_truth = cv2.normalize(ground_truth, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
	
	SSIM_b = ssim(result[:,:,0], ground_truth[:,:,0])
	SSIM_g = ssim(result[:,:,1], ground_truth[:,:,1])
	SSIM_r = ssim(result[:,:,2], ground_truth[:,:,2])
	SSIM = (SSIM_b+SSIM_g+SSIM_r)/3.0

	# print(SSIM)
	MSE = np.mean((ground_truth - result)**2)
	PSNR = 20*np.log10(np.max(ground_truth)/np.sqrt(MSE))
	print(PSNR)
	return result

def interactive_value(blurred_image, kernel):
    D_min = 1000
    D_max = 99859
    D_init = 700
    fig = plt.figure()
    plt.axis("off")
    recovered_image = wiener_filter(blurred_image,kernel, D_init)
    recovered_image = cv2.normalize(recovered_image, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
    # recovered_image_plot = plt.imshow(recovered_image)
    recovered_image_plot = plt.imshow(cv2.cvtColor(recovered_image.astype(np.float32), cv2.COLOR_BGR2RGB))
    slider_ax = plt.axes([0.1, 0.05, 0.8, 0.05])
    value_slider = Slider(slider_ax, 'value', D_min, D_max, valinit=D_init)
     # D0_slider.on_changed(update)
    def update(value):
        recovered_image = wiener_filter(blurred_image, kernel, value).astype(np.float32)
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
	kernel = cv2.resize(kernel, (21,21,))
	interactive_value(blurred_image, kernel)