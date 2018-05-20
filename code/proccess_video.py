from moviepy.editor import VideoFileClip
from handle_image import handle_image

output_video = '../test_video_output.mp4'
input_video  = '../test_video.mp4'

clip1 = VideoFileClip(input_video)
out_obj = clip1.fl_image(handle_image)
out_obj.write_videofile( output_video , audio = False)	
