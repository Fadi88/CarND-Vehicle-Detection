from obj_detect_aux import normalize_image
import matplotlib as plt
import pickle
import glob
import cv2

test_images = glob.glob("../test_images/*.jpg")

for image_file in test_images:
	
	img      = cv2.imread(image_file)
	img_norm = normalize_image(img)
	
	file_name = image_file[image_file.rfind('/')+1 :]
	#window for 64 pixel
	img = cv2.rectangle(img , (680,390) , (900,460) , (255,0,0) , 1)
	img = cv2.rectangle(img , (680,390) , (680+64,390+64) , (255,255,0) , 1)
	#window for 96 pixel
	img = cv2.rectangle(img , (588,400) , (972,480) , (0,0,255) , 1)
	#window for 128 pixel
	img = cv2.rectangle(img , (588,420) , (1280,570) , (0,255,0) , 1)
	cv2.imwrite("../output_images/"+ file_name, img)

img[200:500,300:600,:] = (0,0,0)


cv2.imwrite("test2.jpg" , img)
