import numpy as np
import cv2
from scipy import signal
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim
from skimage.measure import structural_similarity as ssim

# this defined function for ssim was not working as expected so i used in built function 
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
# here kernel size and image sizes are made equal and then basic steps of frequency domain filtering is done
def inverse_filter(blurred_image, kernel):
	rows, cols, channel = blurred_image.shape
	# padding of original image
	pad_blurred_image = np.zeros((2*rows, 2*cols, 3), np.float64)
	pad_blurred_image[0:blurred_image.shape[0], 0:blurred_image.shape[1], :] = blurred_image

	# padding of kernel
	pad_kernel = np.zeros((2*rows, 2*cols, 3), np.float64)
	pad_kernel[0:kernel.shape[0], 0:kernel.shape[1], :] = kernel

	FFT_pad_blurred_image = np.zeros((2*rows, 2*cols,3), np.complex128)
	FFT_kernel = np.zeros((2*rows, 2*cols, 3), np.complex128)
	for i in range(3):
		FFT_pad_blurred_image[:,:,i] = np.fft.fft2(pad_blurred_image[:,:,i])
		FFT_pad_blurred_image[:,:,i] = np.fft.fftshift(FFT_pad_blurred_image[:,:,i])
		FFT_kernel[:,:,i] = np.fft.fft2(pad_kernel[:,:,i])
		FFT_kernel[:,:,i] = np.fft.fftshift(FFT_kernel[:,:,i])
	
	FFT_recovered_image = np.zeros((2*rows, 2*cols, 3), np.complex128)
	recovered_image = np.zeros((2*rows, 2*cols, 3), np.float64)
	
	for i in range(3):	
		FFT_recovered_image[:,:,i] = np.divide(FFT_pad_blurred_image[:,:,i], np.abs(FFT_kernel[:,:,i]))
		FFT_recovered_image[:,:,i] = np.fft.ifftshift(FFT_recovered_image[:,:,i])
		recovered_image[:,:,i] = np.abs(np.fft.ifft2(FFT_recovered_image[:,:,i]))

	result = recovered_image[0:rows, 0:cols, :]
	print(result.dtype)
	return result

if __name__ == '__main__':
	
	blurred_image = cv2.imread("Blurry1_4.jpg", 1)
	blurred_image = cv2.normalize(blurred_image, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)

	ground_truth = cv2.imread("GroundTruth1_1_1.jpg", 1)
	ground_truth = cv2.normalize(ground_truth, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
	

	kernel = cv2.imread("blur_kern_4.png", 1)
	# this is total experiment based but reshaping gave better result in this case
	kernel = cv2.resize(kernel, (21,21, ))
	
	recovered_image = inverse_filter(blurred_image, kernel)
	recovered_image = cv2.normalize(recovered_image, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)

	SSIM_b = ssim(recovered_image[:,:,0], ground_truth[:,:,0])
	SSIM_g = ssim(recovered_image[:,:,1], ground_truth[:,:,1])
	SSIM_r = ssim(recovered_image[:,:,2], ground_truth[:,:,2])
	SSIM = (SSIM_b+SSIM_g+SSIM_r)/3.0
	print(SSIM)
	# because cv imports images as BGR and matplotlib takes images as RGB so to avoid colour inversion colour space is changed 
	recovered_image = cv2.cvtColor(recovered_image, cv2.COLOR_BGR2RGB)
	fig = plt.figure()
	plt.axis("off")
	plt.imshow(recovered_image)
	plt.show()
