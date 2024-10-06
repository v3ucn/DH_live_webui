
import time
import io, os, sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

from pydub import AudioSegment
import numpy as np
from flask import Flask, request, Response,send_from_directory
import torch

import torchaudio
import ffmpeg

import subprocess

from flask_cors import CORS
from flask import make_response,redirect

import json


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


app = Flask(__name__)

CORS(app, cors_allowed_origins="*")

CORS(app, supports_credentials=True)

@app.route("/test", methods=['GET'])
def sft_test():
   
    return {"msg": "ok"}, 200

@app.route("/", methods=['POST'])
def sft_post():
    question_data = request.get_json()

    face = question_data.get('face')
    wav = question_data.get('wav')

    if not face:
        return {"error": "face不能为空"}, 400

    if not wav:
        return {"error": "wav不能为空"}, 400

    convert_wav(wav)

    cmd = fr".\py311\python.exe demo.py video_data/{face} new.wav file/1.mp4"
    print(cmd)
    res = subprocess.Popen(cmd)
    res.wait()


    return {"video":"/file/1.mp4"}, 200




    # # 读取视频文件，使用流式传输
    # def generate():
    #     with open("file/1.mp4", "rb") as f:
    #         while True:
    #             chunk = f.read(4096)  # 读取4KB数据
    #             if not chunk:
    #                 break
    #             yield chunk
    
    # # 返回视频数据
    # return Response(generate(), mimetype="video/mp4",content_type='video/mp4', direct_passthrough=True)
    


    

    


@app.route("/", methods=['GET'])
def sft_get():

    face = request.args.get('face')
    wav = request.args.get('wav')
    
    if not face:
        return {"error": "face不能为空"}, 400

    if not wav:
        return {"error": "wav不能为空"}, 400

    convert_wav(wav)

    cmd = fr".\py311\python.exe demo.py video_data/{face} new.wav file/1.mp4"
    print(cmd)
    res = subprocess.Popen(cmd)
    res.wait()

    return redirect("/file/1.mp4")

    # # 读取视频文件，使用流式传输
    # def generate():
    #     with open("file/1.mp4", "rb") as f:
    #         while True:
    #             chunk = f.read(4096)  # 读取4KB数据
    #             if not chunk:
    #                 break
    #             yield chunk
    
    # # 返回视频数据
    # return Response(generate(), mimetype="video/mp4")

    


@app.route('/file/<filename>')
def uploaded_file(filename):
    return send_from_directory("file", filename)
    

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9880)
