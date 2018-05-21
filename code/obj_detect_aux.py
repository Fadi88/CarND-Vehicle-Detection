from skimage.feature import hog
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import cv2


def extract_hist_feature(img):
	rhist = np.histogram(img[:,:,0] , bins = 32)# , range=(0, 256) )
	ghist = np.histogram(img[:,:,1] , bins = 32)# , range=(0, 256) )
	bhist = np.histogram(img[:,:,2] , bins = 32)# , range=(0, 256) )

	return (rhist[0], ghist[0], bhist[0]) , rhist[1]

def extract_hog_feature(img, orient, pix_per_cell, cell_per_block, vis=True, feature_vec=True):
	return hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),#block_norm= 'L2-Hys',
                                  cells_per_block=(cell_per_block, cell_per_block),transform_sqrt=False, 
                                  visualise= vis, feature_vector= feature_vec)
def extract_spatial_feature(img):
	c1 = cv2.resize(img[:,:,0] , (32,32)).ravel()
	c2 = cv2.resize(img[:,:,1] , (32,32)).ravel()
	c3 = cv2.resize(img[:,:,2] , (32,32)).ravel() 
	return np.hstack( (c1 , c2 , c3) )

def treat_training_image(img_file , debug = False , file_param = True):
	if file_param == True:	
		image = cv2.imread(img_file)
	else :
		image = img_file

	feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
	hist_feature , hist_centers= extract_hist_feature(feature_image)
	
	hog_features = []
	for i in range(3):
		hog_feature, hog_image = extract_hog_feature(feature_image[:,:,i], orient= 9, 
                        		pix_per_cell= 8, cell_per_block= 2,
                        		vis=True, feature_vec=True)
		hog_features.append(hog_feature)

	hog_features = np.ravel( hog_features )
	spatial_features = extract_spatial_feature(feature_image)

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

img = cv2.imread("../training_samples/vehicles/GTI_Far/image0001.png")

