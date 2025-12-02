from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import os
import uuid

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "data"
OUTPUT_FOLDER = "output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/infer', methods=['POST'])
def infer():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio"}), 400
    
    audio_file = request.files['audio']
    audio_id = str(uuid.uuid4())
    audio_path = f"{UPLOAD_FOLDER}/{audio_id}.wav"
    audio_file.save(audio_path)
    
    # 现在先返回一个假视频（等队友做好 inference 再改）
    fake_video = "output/demo.mp4"
    
    # TODO: 队友做好后改成下面这行
    # from inference import run_inference
    # result = run_inference(audio_path)
    # return send_file(result, mimetype='video/mp4')
    
    return send_file(fake_video, mimetype='video/mp4')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)