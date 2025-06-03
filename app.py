from flask import Flask, render_template, Response, jsonify
import cv2
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# VLMモデルの初期化
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = AutoModelForVision2Seq.from_pretrained("Salesforce/blip2-opt-2.7b")

def generate_camera_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # フレームをJPEGに変換
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def analyze_image(image):
    # OpenCVのBGR形式からPIL Imageに変換
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)
    
    # 画像の分析
    inputs = processor(pil_image, return_tensors="pt")
    generated_ids = model.generate(
        pixel_values=inputs.pixel_values,
        max_length=50,
        num_beams=5,
        return_dict_in_generate=True
    )
    generated_text = processor.batch_decode(generated_ids.sequences, skip_special_tokens=True)[0]
    
    return generated_text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_camera_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/analyze_frame')
def analyze_frame():
    camera = cv2.VideoCapture(1)
    if not camera.isOpened():
        return jsonify({'text': 'カメラが開けませんでした'})
    success, frame = camera.read()
    camera.release()
    
    if success:
        analysis = analyze_image(frame)
        return jsonify({'text': analysis})
    return jsonify({'text': 'Error capturing frame'})

if __name__ == '__main__':
    app.run(debug=True) 