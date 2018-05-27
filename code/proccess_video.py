from moviepy.editor import VideoFileClip
from handle_image_module import *

import pickle

test = False
#test = True

if test:
	input_video  = '../test_video.mp4'
	output_video = '../test_video_output.mp4'
else:
	input_video  = '../project_video.mp4'
	output_video = '../project_video_output.mp4'


def process_frame(frame):
	global clf

	return find_cars(frame)

clip1 = VideoFileClip(input_video)
out_obj = clip1.fl_image(process_frame)
out_obj.write_videofile( output_video , audio = False)

