import time
import os
import numpy as np
import uuid
import cv2
import tqdm
import shutil
import sys
from talkingface.audio_model import AudioModel
from talkingface.render_model import RenderModel
import torch


miss_list = ['source_in_conv.0.conv.weight', 'source_in_conv.0.conv.bias', 'source_in_conv.0.norm.weight', 'source_in_conv.0.norm.bias', 'source_in_conv.0.norm.running_mean', 'source_in_conv.0.norm.running_var', 'source_in_conv.0.norm.num_batches_tracked', 'source_in_conv.1.conv.weight', 'source_in_conv.1.conv.bias', 'source_in_conv.1.norm.weight', 'source_in_conv.1.norm.bias', 'source_in_conv.1.norm.running_mean', 'source_in_conv.1.norm.running_var', 'source_in_conv.1.norm.num_batches_tracked', 'source_in_conv.2.conv.weight', 'source_in_conv.2.conv.bias', 'source_in_conv.2.norm.weight', 'source_in_conv.2.norm.bias', 'source_in_conv.2.norm.running_mean', 'source_in_conv.2.norm.running_var', 'source_in_conv.2.norm.num_batches_tracked', 'ref_in_conv.0.conv.weight', 'ref_in_conv.0.conv.bias', 'ref_in_conv.0.norm.weight', 'ref_in_conv.0.norm.bias', 'ref_in_conv.0.norm.running_mean', 'ref_in_conv.0.norm.running_var', 'ref_in_conv.0.norm.num_batches_tracked', 'ref_in_conv.1.conv.weight', 'ref_in_conv.1.conv.bias', 'ref_in_conv.1.norm.weight', 'ref_in_conv.1.norm.bias', 'ref_in_conv.1.norm.running_mean', 'ref_in_conv.1.norm.running_var', 'ref_in_conv.1.norm.num_batches_tracked', 'ref_in_conv.2.conv.weight', 'ref_in_conv.2.conv.bias', 'ref_in_conv.2.norm.weight', 'ref_in_conv.2.norm.bias', 'ref_in_conv.2.norm.running_mean', 'ref_in_conv.2.norm.running_var', 'ref_in_conv.2.norm.num_batches_tracked', 'trans_conv.0.conv.weight', 'trans_conv.0.conv.bias', 'trans_conv.0.norm.weight', 'trans_conv.0.norm.bias', 'trans_conv.0.norm.running_mean', 'trans_conv.0.norm.running_var', 'trans_conv.0.norm.num_batches_tracked', 'trans_conv.1.conv.weight', 'trans_conv.1.conv.bias', 'trans_conv.1.norm.weight', 'trans_conv.1.norm.bias', 'trans_conv.1.norm.running_mean', 'trans_conv.1.norm.running_var', 'trans_conv.1.norm.num_batches_tracked', 'trans_conv.2.conv.weight', 'trans_conv.2.conv.bias', 'trans_conv.2.norm.weight', 'trans_conv.2.norm.bias', 'trans_conv.2.norm.running_mean', 'trans_conv.2.norm.running_var', 'trans_conv.2.norm.num_batches_tracked', 'trans_conv.3.conv.weight', 'trans_conv.3.conv.bias', 'trans_conv.3.norm.weight', 'trans_conv.3.norm.bias', 'trans_conv.3.norm.running_mean', 'trans_conv.3.norm.running_var', 'trans_conv.3.norm.num_batches_tracked', 'trans_conv.4.conv.weight', 'trans_conv.4.conv.bias', 'trans_conv.4.norm.weight', 'trans_conv.4.norm.bias', 'trans_conv.4.norm.running_mean', 'trans_conv.4.norm.running_var', 'trans_conv.4.norm.num_batches_tracked', 'trans_conv.5.conv.weight', 'trans_conv.5.conv.bias', 'trans_conv.5.norm.weight', 'trans_conv.5.norm.bias', 'trans_conv.5.norm.running_mean', 'trans_conv.5.norm.running_var', 'trans_conv.5.norm.num_batches_tracked', 'trans_conv.6.conv.weight', 'trans_conv.6.conv.bias', 'trans_conv.6.norm.weight', 'trans_conv.6.norm.bias', 'trans_conv.6.norm.running_mean', 'trans_conv.6.norm.running_var', 'trans_conv.6.norm.num_batches_tracked', 'trans_conv.7.conv.weight', 'trans_conv.7.conv.bias', 'trans_conv.7.norm.weight', 'trans_conv.7.norm.bias', 'trans_conv.7.norm.running_mean', 'trans_conv.7.norm.running_var', 'trans_conv.7.norm.num_batches_tracked', 'trans_conv.8.conv.weight', 'trans_conv.8.conv.bias', 'trans_conv.8.norm.weight', 'trans_conv.8.norm.bias', 'trans_conv.8.norm.running_mean', 'trans_conv.8.norm.running_var', 'trans_conv.8.norm.num_batches_tracked', 'appearance_conv_list.0.0.conv1.weight', 'appearance_conv_list.0.0.conv1.bias', 'appearance_conv_list.0.0.conv2.weight', 'appearance_conv_list.0.0.conv2.bias', 'appearance_conv_list.0.0.norm1.weight', 'appearance_conv_list.0.0.norm1.bias', 'appearance_conv_list.0.0.norm1.running_mean', 'appearance_conv_list.0.0.norm1.running_var', 'appearance_conv_list.0.0.norm1.num_batches_tracked', 'appearance_conv_list.0.0.norm2.weight', 'appearance_conv_list.0.0.norm2.bias', 'appearance_conv_list.0.0.norm2.running_mean', 'appearance_conv_list.0.0.norm2.running_var', 'appearance_conv_list.0.0.norm2.num_batches_tracked', 'appearance_conv_list.0.1.conv1.weight', 'appearance_conv_list.0.1.conv1.bias', 'appearance_conv_list.0.1.conv2.weight', 'appearance_conv_list.0.1.conv2.bias', 'appearance_conv_list.0.1.norm1.weight', 'appearance_conv_list.0.1.norm1.bias', 'appearance_conv_list.0.1.norm1.running_mean', 'appearance_conv_list.0.1.norm1.running_var', 'appearance_conv_list.0.1.norm1.num_batches_tracked', 'appearance_conv_list.0.1.norm2.weight', 'appearance_conv_list.0.1.norm2.bias', 'appearance_conv_list.0.1.norm2.running_mean', 'appearance_conv_list.0.1.norm2.running_var', 'appearance_conv_list.0.1.norm2.num_batches_tracked', 'appearance_conv_list.1.0.conv1.weight', 'appearance_conv_list.1.0.conv1.bias', 'appearance_conv_list.1.0.conv2.weight', 'appearance_conv_list.1.0.conv2.bias', 'appearance_conv_list.1.0.norm1.weight', 'appearance_conv_list.1.0.norm1.bias', 'appearance_conv_list.1.0.norm1.running_mean', 'appearance_conv_list.1.0.norm1.running_var', 'appearance_conv_list.1.0.norm1.num_batches_tracked', 'appearance_conv_list.1.0.norm2.weight', 'appearance_conv_list.1.0.norm2.bias', 'appearance_conv_list.1.0.norm2.running_mean', 'appearance_conv_list.1.0.norm2.running_var', 'appearance_conv_list.1.0.norm2.num_batches_tracked', 'appearance_conv_list.1.1.conv1.weight', 'appearance_conv_list.1.1.conv1.bias', 'appearance_conv_list.1.1.conv2.weight', 'appearance_conv_list.1.1.conv2.bias', 'appearance_conv_list.1.1.norm1.weight', 'appearance_conv_list.1.1.norm1.bias', 'appearance_conv_list.1.1.norm1.running_mean', 'appearance_conv_list.1.1.norm1.running_var', 'appearance_conv_list.1.1.norm1.num_batches_tracked', 'appearance_conv_list.1.1.norm2.weight', 'appearance_conv_list.1.1.norm2.bias', 'appearance_conv_list.1.1.norm2.running_mean', 'appearance_conv_list.1.1.norm2.running_var', 'appearance_conv_list.1.1.norm2.num_batches_tracked', 'adaAT.commn_linear.0.weight', 'adaAT.commn_linear.0.bias', 'adaAT.scale.0.weight', 'adaAT.scale.0.bias', 'adaAT.rotation.0.weight', 'adaAT.rotation.0.bias', 'adaAT.translation.0.weight', 'adaAT.translation.0.bias', 'out_conv.0.conv.weight', 'out_conv.0.conv.bias', 'out_conv.0.norm.weight', 'out_conv.0.norm.bias', 'out_conv.0.norm.running_mean', 'out_conv.0.norm.running_var', 'out_conv.0.norm.num_batches_tracked', 'out_conv.1.conv.weight', 'out_conv.1.conv.bias', 'out_conv.1.norm.weight', 'out_conv.1.norm.bias', 'out_conv.1.norm.running_mean', 'out_conv.1.norm.running_var', 'out_conv.1.norm.num_batches_tracked', 'out_conv.2.conv1.weight', 'out_conv.2.conv1.bias', 'out_conv.2.conv2.weight', 'out_conv.2.conv2.bias', 'out_conv.2.norm1.weight', 'out_conv.2.norm1.bias', 'out_conv.2.norm1.running_mean', 'out_conv.2.norm1.running_var', 'out_conv.2.norm1.num_batches_tracked', 'out_conv.2.norm2.weight', 'out_conv.2.norm2.bias', 'out_conv.2.norm2.running_mean', 'out_conv.2.norm2.running_var', 'out_conv.2.norm2.num_batches_tracked', 'out_conv.3.conv.weight', 'out_conv.3.conv.bias', 'out_conv.3.norm.weight', 'out_conv.3.norm.bias', 'out_conv.3.norm.running_mean', 'out_conv.3.norm.running_var', 'out_conv.3.norm.num_batches_tracked', 'out_conv.4.weight', 'out_conv.4.bias']

def main():
    # 检查命令行参数的数量
    if len(sys.argv) != 4:
        print("Usage: python demo.py <video_path> <output_video_name>")
        sys.exit(1)  # 参数数量不正确时退出程序

    # 获取video_name参数
    video_path = sys.argv[1]
    print(f"Video path is set to: {video_path}")
    audio_path = sys.argv[2]
    print(f"Audio path is set to: {audio_path}")
    output_video_name = sys.argv[3]
    print(f"output video name is set to: {output_video_name}")

    audioModel = AudioModel()
    audioModel.loadModel("checkpoint/audio.pkl")

    renderModel = RenderModel()
    renderModel.loadModel("checkpoint/render.pth")


    pkl_path = "{}/keypoint_rotate.pkl".format(video_path)
    video_path = "{}/circle.mp4".format(video_path)
    renderModel.reset_charactor(video_path, pkl_path)

    # wavpath = "video_data/audio0.wav"
    wavpath = audio_path
    mouth_frame = audioModel.interface_wav(wavpath)
    cap_input = cv2.VideoCapture(video_path)
    vid_width = cap_input.get(cv2.CAP_PROP_FRAME_WIDTH)  # 宽度
    vid_height = cap_input.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 高度
    cap_input.release()
    task_id = str(uuid.uuid1())
    os.makedirs("output/{}".format(task_id), exist_ok=True)
    # 定义编码器
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    save_path = "output/{}/silence.mp4".format(task_id)
    videoWriter = cv2.VideoWriter(save_path, fourcc, 25, (int(vid_width) * 1, int(vid_height)))
    for frame in tqdm.tqdm(mouth_frame):
        frame = renderModel.interface(frame)
        # cv2.imshow("s", frame)
        # cv2.waitKey(40)

        videoWriter.write(frame)

    videoWriter.release()
    val_video = "../output/{}.mp4".format(task_id)
    os.system("ffmpeg -y -i {} -i {} -c:v copy -pix_fmt yuv420p -loglevel quiet {}".format(save_path, wavpath, output_video_name))
    # os.system("ffmpeg -y -i {} -i {} -c:v libx264 -c:a copy -pix_fmt yuv420p -loglevel quiet {}".format(save_path, wavpath, output_video_name))
    shutil.rmtree("output/{}".format(task_id))



if __name__ == "__main__":
    main()