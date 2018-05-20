from skimage.feature import hog
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2

def normalize_image(img):
	if np.max(img) > 1:
		img = (img - np.average(img)) / (np.max(img) - np.min(img))
	return img
		
def extract_hist_feature(img):
	img = cv2.cvtColor(img , cv2.COLOR_BGR2YCrCb)
	#images are BGR format since they are read by open CV
	rhist = np.histogram(img[:,:,2] , bins = 32 , range=(0, 256) )
	ghist = np.histogram(img[:,:,1] , bins = 32 , range=(0, 256) )
	bhist = np.histogram(img[:,:,0] , bins = 32 , range=(0, 256) )

	return (rhist[0], ghist[0], bhist[0]) , rhist[1]

def extract_hog_feature(img, orient, pix_per_cell, cell_per_block, vis=True, feature_vec=True):
	return hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  block_norm= 'L2-Hys', transform_sqrt=False, 
                                  visualise= vis, feature_vector= feature_vec)
def extract_spatial_feature(img):
	color_trsf_img = cv2.cvtColor(img , cv2.COLOR_BGR2YCrCb)	
	return cv2.resize(color_trsf_img , (32,32)).ravel()

def treat_training_image(img_file , debug = False , file_param = True):
	if file_param == True:	
		image = cv2.imread(img_file)
	else :
		image = img_file

	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	#image_norm = normalize_image(image)

	hist_feature , hist_centers= extract_hist_feature(image)

	hog_features, hog_image = extract_hog_feature(gray, orient= 11, 
                        	pix_per_cell= 8, cell_per_block= 2, 
                        	vis=True, feature_vec=True)

	spatial_features = extract_spatial_feature(image)

	if debug == True :
		file_name = img_file[img_file.rfind('/') + 1 :]
		
		bin_edges = hist_centers
		bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2

		fig = plt.figure(figsize=(24 , 3))
		plt.subplot(131)
		plt.bar(bin_centers, hist_feature[0])
		plt.xlim(-1, 1)
		plt.title('R Histogram')
		plt.subplot(132)
		plt.bar(bin_centers, hist_feature[1])
		plt.xlim(-1, 1)
		plt.title('G Histogram')
		plt.subplot(133)
		plt.bar(bin_centers, hist_feature[2])
		plt.xlim(-1, 1)
		plt.title('B Histogram')

		fig.savefig("../debug/hist_" + file_name )
		
		fig = plt.figure(figsize=(20 , 10)) 
		
		plt.imshow(hog_image , cmap="gray" )

		fig.savefig("../debug/hog_" + file_name )

	hist = np.concatenate((hist_feature[0] , hist_feature[1] , hist_feature[2]))	
	hist =  (hist  - np.average(hist) ) / (np.max(hist) - np.min(hist) )

	hog_features = (hog_features - np.average(hog_features)) / (np.max(hog_features) - np.min(hog_features))
	
	
	spatial_features = (spatial_features-np.average(spatial_features))/(np.max(spatial_features) -np.min(spatial_features))
	
	
	feature_vector = np.concatenate(( hist , hog_features , spatial_features))

	return feature_vector

t0 = time.time()
tmp = treat_training_image("../test_images/test1.jpg" , True)
print("time for a single image : " , time.time() - t0)
print("single feature vector is of length : " , len(tmp))

img = cv2.imread("../training_samples/vehicles/GTI_Far/image0001.png")

print("max read value for training is : " , np.max(img))

