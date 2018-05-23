from obj_detect_aux import treat_training_image
from scipy.ndimage.measurements import label

import matplotlib as plt
import numpy as np
import pickle
import glob
import cv2


def handle_image(img , search_boxes , clf , heat_thres = 1):
	flt_heatmap = gen_filtered_heat_map(img , search_boxes , clf , heat_thres)
	cv2.imwrite('../debug/heat_map.jpg' , flt_heatmap)

	labels = label(flt_heatmap)
	labeled_heatmaps = labels[0]
	
	for car_number in range(1, labels[1]+1):
		# Find pixels with each car_number label value
		nonzero = (labels[0] == car_number).nonzero()
		# Identify x and y values of those pixels
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		# Define a bounding box based on min/max x and y
		bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
		# Draw the box on the image
		cv2.rectangle(img, bbox[0], bbox[1], (255,0,0), 2)
	# Return the image
	return img

def gen_filtered_heat_map(img , search_boxes , clf , heat_thres = 1):

	heat_map = np.zeros_like(img[:,:,0])

	for search_box in search_boxes:
		upper_left  = search_box[0]
		lower_right = search_box[1]

		size  = lower_right[0] - upper_left[0]

		patch = img[  upper_left[1] : lower_right [1]  , upper_left[0] : lower_right [0]  , :]

				
		if size != 64:
			patch = cv2.resize(patch , (64 , 64))
		
		if clf.predict(treat_training_image(patch , debug=False , file_param = False).reshape(1, -1)) == 1:
			heat_map[upper_left[1] : lower_right [1]  , upper_left[0] : lower_right [0] ] += 1 	



	return ((heat_map > heat_thres)*255).astype(np.uint8)

test_images = glob.glob("../test_images/*.jpg")



search_zones = []

search_zone = {}

search_zone['box_size'] = 64
search_zone['scale'] = 1
search_zone['upper_left']  = (500,390)
search_zone['lower_right'] = (1280,520)

#search_zones.append(search_zone)

search_zone = {}

search_zone['box_size'] = 96
search_zone['scale'] = 1.5
search_zone['upper_left']  = (650,400)
search_zone['lower_right'] = (1280,656)

search_zones.append(search_zone)

search_zone = {}

search_zone['box_size'] = 128
search_zone['scale'] = 2
search_zone['upper_left']  = (400,400)
search_zone['lower_right'] = (1280,656)

#search_zones.append(search_zone)

x_step = 16
y_step = 16

search_boxes = []

for zone in search_zones:
	box_size = zone['box_size']
	scale = zone['scale']
	upper_left = zone['upper_left']
	lower_right = zone['lower_right']

	for y_current in range(upper_left[1] , lower_right[1] - box_size , y_step):	
		for x_current in range(upper_left[0] , lower_right[0] - box_size  , x_step):
			search_boxes.append( ((x_current , y_current) , (x_current + box_size , y_current + box_size) ))

print("boxes count is : " , len(search_boxes))

clf = pickle.load(open("clf.p" , "rb"))

pickle.dump(search_boxes , open("search_boxes.p","wb"))

for img_file in test_images:

	img = cv2.imread(img_file)	
	cv2.imwrite( img_file.replace('test_images' , 'output_images'), handle_image(img , search_boxes , clf))



