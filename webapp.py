from webui_dh_live import app


if __name__ == '__main__':
    app.queue()
    app.launch(inbrowser=True)