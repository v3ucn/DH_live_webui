import argparse
import os

# 设置HF_ENDPOINT环境变量
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 设置HF_HOME环境变量为当前目录下的hf_download文件夹
os.environ["HF_HOME"] = os.path.join(os.getcwd(), "hf_download")
import gradio as gr
import subprocess

from PIL import Image
import numpy as np
from pydub import AudioSegment

from en_test import face_en
import requests

initial_md = """

官方项目地址：https://github.com/kleinlee/DH_live

整合包制作:刘悦的技术博客 https://space.bilibili.com/3031494

"""

def do_face_1(face):

    filename = os.path.basename(face)

    face_en(face,f"{filename}_sr.mp4")

    return f"{filename}_sr.mp4"

def do_face_2(face):

    filename = os.path.basename(face)

    face_en(face,f"1_sr.mp4")

    return f"1_sr.mp4"

reference_wavs = ["请选择视频监测点目录"]
for name in os.listdir("./video_data/"):
    reference_wavs.append(name)

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

def change_choices():

    reference_wavs = ["请选择视频监测点目录"]

    for name in os.listdir("./video_data/"):
        reference_wavs.append(name)
    
    return {"choices":reference_wavs, "__type__": "update"}

def do_make(face):

    filename = os.path.basename(face)

    source_file = face  # 请替换为您的源文件路径
    # 目标文件路径
    destination_file = filename

    # 使用 with 语句打开源文件和目标文件
    with open(source_file, "rb") as source, open(destination_file, "wb") as destination:
        # 读取源文件的所有内容
        data = source.read()
        # 将内容写入目标文件
        destination.write(data)

    cmd = fr".\py311\python.exe data_preparation.py {filename}"   
    print(cmd)

    res = subprocess.Popen(cmd)
    res.wait()

    return "生成完成"

def do_cloth(face,audio,model_name):

    convert_wav(audio)

    cmd = fr".\py311\python.exe demo.py video_data/{face} new.wav 1.mp4 {model_name}"

    print(cmd)
    res = subprocess.Popen(cmd)
    res.wait()
    
    return "1.mp4"


# 请求接口
def request_api(url):

    print(url)

    response = requests.get(url)
    audio_data = response.content

    with open(f"api_audio.wav","wb") as f:
        f.write(audio_data)
    
    return "./api_audio.wav"

with gr.Blocks() as app:
    gr.Markdown(initial_md)

    with gr.Accordion("素材制作"):
        with gr.Row():
            video = gr.Video(label="上传视频")
            result_text = gr.Textbox(label="检查点生成结果")
            # first_button = gr.Button("面部超分")
            make_button = gr.Button("生成检查点文件")

    with gr.Accordion("素材选择"):
        with gr.Row():

            face = gr.Dropdown(label="人脸目录列表",choices=reference_wavs,interactive=True)
            refresh_button = gr.Button("刷新人脸目录列表")
            refresh_button.click(fn=change_choices, inputs=[], outputs=[face])

            uploaded_audio = gr.Audio(type="filepath", label="驱动音频,采样率无所谓,会自动转16kHz采样率、16位bit的单声道")
            
            output_video = gr.Video()

        with gr.Row():

            api_url = gr.Textbox(label="输入接口地址和参数,可以是gpt-sovits,也可以是别的TTS项目接口", lines=4,value='http://127.0.0.1:9880/?text=我问她月饼爱吃咸的还是甜的，那天老色鬼说,你身上的月饼，自然是甜过了蜜糖。&text_lang=zh&ref_audio_path=./参考音频/[jok老师]说得好像您带我以来我考好过几次一样.wav&prompt_lang=zh&prompt_text=说得好像您带我以来我考好过几次一样&text_split_method=cut5')


            api_button = gr.Button("请求音频接口")
            api_button.click(fn=request_api, inputs=[api_url], outputs=[uploaded_audio])

        with gr.Row():
            model_name = gr.Textbox(label="推理使用的模型",value="render.pth")
            generate_button = gr.Button("生成数字人视频")
            second_button = gr.Button("视频结果面部超分")

    # with gr.Accordion("音频转换(16kHz采样率、16位bit的单声道)"):
    #     with gr.Row():

    #         old = gr.Audio(type="filepath", label="请上传需要转换音频文件")
    #         audio_button = gr.Button("音频转换")
    #         new = gr.Audio()

    #         audio_button.click(convert_wav,inputs=[old],outputs=[new])
            

            
            

    generate_button.click(do_cloth,inputs=[face,uploaded_audio,model_name],outputs=[output_video])
    make_button.click(do_make,inputs=[video],outputs=[result_text])
    # first_button.click(do_face_1,inputs=[video],outputs=[video])
    second_button.click(do_face_2,inputs=[output_video],outputs=[output_video])
    
if __name__ == '__main__':
    app.queue()
    app.launch(inbrowser=True)

    


