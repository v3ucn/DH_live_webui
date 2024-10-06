import requests
import json

headers = {'Content-Type': 'application/json'}

gpt = {"face":"gakki.mp4","wav":"keira.wav"}

response = requests.post("http://localhost:9880/",data=json.dumps(gpt),headers=headers)

audio_data = response.json()

print(audio_data)