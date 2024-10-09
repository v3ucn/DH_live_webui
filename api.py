
import time
import io, os, sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

from pydub import AudioSegment
import numpy as np
from flask import Flask, request, Response,send_from_directory,render_template
import torch

import torchaudio
import ffmpeg

from flask_cors import CORS
from flask import make_response,redirect

import json

import platform

from subprocess import Popen

import subprocess

import edge_tts

import asyncio

ps1a=[]


system=platform.system()
def kill_process(pid):
    if(system=="Windows"):
        cmd = "taskkill /t /f /pid %s" % pid
        os.system(cmd)
    else:
        kill_proc_tree(pid)


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


app = Flask(__name__,static_folder='static', static_url_path='/static')

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


    # return {"video":"/file/1.mp4"}, 200

    # 读取视频文件，使用流式传输
    def generate():
        with open("file/1.mp4", "rb") as f:
            while True:
                chunk = f.read(4096)  # 读取4KB数据
                if not chunk:
                    break
                yield chunk
    
    # 返回视频数据
    return Response(generate(), mimetype="video/mp4",content_type='video/mp4', direct_passthrough=True)
    

@app.route("/push_talk", methods=['GET'])
def push_talk():

    global ps1a

    for p_asr in ps1a:
        kill_process(p_asr.pid)

    face = request.args.get('face')
    
    # if not face_video:
    #     return {"error": "face不能为空"}, 400

    # cmd = fr"ffmpeg -stream_loop -1 -i quiet_output/gakki_quiet.mp4 -vcodec copy -acodec copy -f flv -y rtmp://localhost/live/livestream"
    # print(cmd)
    # res = subprocess.Popen(cmd)
    # res.wait()

    cmd = f"ffmpeg -stream_loop -1 -i {face} -vf scale=350:640 -b:v 200k -vcodec libx264 -acodec copy -f flv -y rtmp://localhost/live/livestream"
    p = Popen(cmd, shell=True)
    p.wait()

    
    cmd = f"ffmpeg -stream_loop -1 -i quiet_output/gakki.mp4 -vf scale=350:640 -b:v 200k -vcodec libx264 -acodec copy -f flv -y rtmp://localhost/live/livestream"
    p = Popen(cmd, shell=True)
    ps1a.append(p)

    return {"msg": "ok"}, 200

@app.route("/push_quiet", methods=['GET'])
def push_quiet():

    global ps1a

    for p_asr in ps1a:
        kill_process(p_asr.pid)

    face = request.args.get('face')
    
    
    # if not face_video:
    #     return {"error": "face不能为空"}, 400

    # cmd = fr"ffmpeg -stream_loop -1 -i quiet_output/gakki_quiet.mp4 -vcodec copy -acodec copy -f flv -y rtmp://localhost/live/livestream"
    # print(cmd)
    # res = subprocess.Popen(cmd)
    # res.wait()

    cmd = f"ffmpeg -stream_loop -1 -i quiet_output/{face} -vf scale=350:640 -b:v 200k -vcodec libx264 -acodec copy -f flv -y rtmp://localhost/live/livestream"
    p = Popen(cmd, shell=True)
    ps1a.append(p)

    return {"msg": "ok"}, 200


async def amain(TEXT) -> None:
    """Main function"""
    communicate = edge_tts.Communicate(TEXT, "zh-CN-XiaoxiaoNeural")
    await communicate.save("edge_tts.wav")
@app.route("/make_video", methods=['GET'])
def make_video():

    face = request.args.get('face')
    text = request.args.get('text')

    asyncio.run(amain(text))

    # convert_wav("edge_tts.wav")

    cmd = fr"ffmpeg -i edge_tts.wav -acodec pcm_s16le -ac 1 -ar 16000 -y new.wav"
    print(cmd)
    res = subprocess.Popen(cmd)
    res.wait()

    cmd = fr".\py311\python.exe demo.py video_data/{face} new.wav static/1.mp4"
    print(cmd)
    res = subprocess.Popen(cmd)
    res.wait()

    # if not face_video:
    #     return {"error": "face不能为空"}, 400

    # cmd = fr"ffmpeg -stream_loop -1 -i quiet_output/gakki_quiet.mp4 -vcodec copy -acodec copy -f flv -y rtmp://localhost/live/livestream"
    # print(cmd)
    # res = subprocess.Popen(cmd)
    # res.wait()


    return {"msg": "ok"}, 200

    


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

    # return redirect("/file/1.mp4")

    # 读取视频文件，使用流式传输
    def generate():
        with open("file/1.mp4", "rb") as f:
            while True:
                chunk = f.read(4096)  # 读取4KB数据
                if not chunk:
                    break
                yield chunk
    
    # 返回视频数据
    return Response(generate(), mimetype="video/mp4")

    

# 网页模版

@app.route('/player')
def player():
    return render_template('player.html')


@app.route('/video')
def video():
    return render_template('video.html')

@app.route('/testhtml')
def testhtml():
    return render_template('test.html')

@app.route('/file/<filename>')
def uploaded_file(filename):
    return send_from_directory("file", filename)
    

if __name__ == "__main__":
    print("实时数字人请访问 http://localhost:9880/video ")
    app.run(host='0.0.0.0',port=9880,debug=True)
