from obj_detect_aux import treat_training_image
from scipy.ndimage.measurements import label
from lesson_function import get_hog_features
from obj_detect_aux import extract_spatial_feature
from obj_detect_aux import extract_hist_feature

import matplotlib as plt
import numpy as np
import pickle
import glob
import cv2

cnt = 0

def find_possible_cars(img, ystart, ystop, scale, svc):

    ret = []

    draw_img = np.copy(img)
    heat_map = np.zeros_like(img[:,:,0])
    ctrans_tosearch = cv2.cvtColor(img[ystart:ystop,:,:], cv2.COLOR_RGB2YCrCb)

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // 8) - 2 + 1
    nyblocks = (ch1.shape[0] // 8) - 2 + 1 
    nfeat_per_block = 9*2**2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // 8) - 2 + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, 9, 8, 2, feature_vec=False)
    hog2 = get_hog_features(ch2, 9, 8, 2, feature_vec=False)
    hog3 = get_hog_features(ch3, 9, 8, 2, feature_vec=False)
    x = 0 

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*2
            xpos = xb*2
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*8
            ytop = ypos*8

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

            # Get color features
            spatial_features = extract_spatial_feature(subimg)
            hist_features_tu  , _ = extract_hist_feature(subimg)
            hist_features = np.concatenate((hist_features_tu[0] , hist_features_tu[1] , hist_features_tu[2]))

            # Scale features and make a prediction
            spatial_features = (spatial_features - np.average(spatial_features)) / (np.max(spatial_features) - np.min(spatial_features))
            hist_features    = (hist_features - np.average(hist_features)) / (np.max(hist_features) - np.min(hist_features))
            hog_features     = (hog_features - np.average(hog_features)) / (np.max(hog_features) - np.min(hog_features))

            test_features    = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)

            #test_prediction = svc.predict(test_features)
            test_prediction = svc.predict([treat_training_image(subimg , file_param = False)])
            global cnt
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                ret.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                cv2.imwrite( '../debug/video/1/' + str(cnt) + '.jpg', cv2.cvtColor(subimg,cv2.COLOR_YCrCb2RGB))
            else:
                cv2.imwrite( '../debug/video/0/' + str(cnt) + '.jpg', cv2.cvtColor(subimg,cv2.COLOR_YCrCb2RGB))
            cnt += 1

    return ret

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




def find_cars(img):
	global clf

	found_Cars = []

	heat_map = np.zeros_like(img[:,:,0])

	#found_Cars.append(find_possible_cars(img , 400 , 656 , 1  , clf))
	found_Cars.append(find_possible_cars(img , 400 , 656 , 1.5  , clf))
	#found_Cars.append(find_possible_cars(img , 400 , 560 , 2  , clf))
	#found_Cars.append(find_possible_cars(img , 400 , 600 , 3  , clf))
	for size in found_Cars :
		for rect in size:
			heat_map[rect[0][1]:rect[1][1] ,rect[0][0]:rect[1][0]] += 1
			#cv2.rectangle(img, rect[0], rect[1], (255,0,0), 2)
	heat_map = ((heat_map > 5)*255).astype(np.uint8)
	
	labels  = label(heat_map )

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

clf = pickle.load(open("clf.p" , "rb"))

for img_file in test_images:

	img = cv2.imread(img_file)	
	cv2.imwrite( img_file.replace('test_images' , 'output_images'), find_cars(img))



