from webui_dh_live_train import app

if __name__ == '__main__':
    app.queue()
    app.launch(inbrowser=True)