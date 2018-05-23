from moviepy.editor import VideoFileClip
from handle_image_module import *

import pickle

input_video  = '../project_video.mp4'
output_video = '../project_video_output.mp4'

clf = pickle.load(open("clf.p" , "rb"))
search_boxes = pickle.load(open("search_boxes.p" , "rb"))

def process_frame(frame):
	global clf
	global search_boxes

	return handle_image(frame , search_boxes , clf , 3)

clip1 = VideoFileClip(input_video)
out_obj = clip1.fl_image(process_frame)
out_obj.write_videofile( output_video , audio = False)

