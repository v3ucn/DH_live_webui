import torch
from pydub import AudioSegment
import numpy as np

def convert_wav(input_file):
    # 加载音频文件
    audio = AudioSegment.from_wav(input_file)
    
    # 转换为单声道
    audio = audio.set_channels(1)
    
    # 设置采样率为16kHz
    audio = audio.set_frame_rate(16000)
    
    # 设置采样位数为16位
    audio = audio.set_sample_width(2)  # 2 bytes = 16 bits
    
    # 导出处理后的音频
    audio.export("new.wav", format="wav")

    return "new.wav"


if __name__ == '__main__':
    
    # convert_wav("edge_tts.wav")


    checkpoint = torch.load("checkpoint/render.pth")

    print(checkpoint.keys())

    # checkpoint = torch.load("checkpoint/epoch-160.pth")

    # print(checkpoint['state_dict']["net_g"].keys())

# print(checkpoint['state_dict']["net_g"].keys())


# print(3333)


# checkpoint = torch.load("checkpoint/render.pth")

# print(checkpoint.keys())

# print(checkpoint['state_dict']["net_g"].keys())


