import sys,os
from face_sr.face_enhancer import enhancer_list
import imageio
from moviepy.editor import VideoFileClip,AudioFileClip



def face_en(old,new):

    predicted_video_512_path = new
    predicted_video_256_path = old

    # Super-resolution
    imageio.mimsave(predicted_video_512_path+'.tmp.mp4', enhancer_list(predicted_video_256_path, method='gfpgan', bg_upsampler=None), fps=float(25))

    # Merge audio and video
    video_clip = VideoFileClip(predicted_video_512_path+'.tmp.mp4')
    audio_clip = AudioFileClip(predicted_video_256_path)
    final_clip = video_clip.set_audio(audio_clip)
    final_clip.write_videofile(predicted_video_512_path, codec='libx264', audio_codec='aac')

    os.remove(predicted_video_512_path+'.tmp.mp4')


if __name__ == '__main__':
    
    face_en("./new1.mp4","./newnew.mp4")
