import argparse
import os

os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1" 

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


def do_pre():

    cmd = fr".\py311\python.exe train/data_preparation_face.py ./train/data "

    print(cmd)
    res = subprocess.Popen(cmd)
    res.wait()
    
    return "数据预处理成功"


def do_lip():

    cmd = fr".\py311\python.exe train/train_input_validation_render_model.py ./train/data "

    print(cmd)
    res = subprocess.Popen(cmd)
    res.wait()
    
    return "唇形检测成功"


def do_train():

    cmd = fr".\py311\python.exe train/train_render_model.py --train_data ./train/data "

    print(cmd)
    res = subprocess.Popen(cmd)
    res.wait()
    
    return "训练完毕"



with gr.Blocks() as app:
    gr.Markdown(initial_md)

    with gr.Accordion("render模型训练"):

        with gr.Row():
            pre_button = gr.Button("数据预处理")
            pre_text = gr.Textbox(label="数据预处理结果")

        with gr.Row():
            lip_button = gr.Button("唇形检测")
            lip_text = gr.Textbox(label="唇形检测处理结果")

        with gr.Row():
            train_button = gr.Button("开始训练")
            train_text = gr.Textbox(label="训练结果")
            


    pre_button.click(do_pre,inputs=[],outputs=[pre_text])

    lip_button.click(do_lip,inputs=[],outputs=[lip_text])

    train_button.click(do_train,inputs=[],outputs=[train_text])
    
if __name__ == '__main__':
    app.queue()
    app.launch(inbrowser=True)

    


