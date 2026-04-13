# 彻底关闭TensorFlow警告
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, send_file, request, jsonify
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# 加载模型（compile=False避免指标警告）
model = tf.keras.models.load_model("mnist_cnn_model.h5", compile=False)

# 首页
@app.route('/')
def index():
    return send_file('index.html')

# 识别接口（核心修复：严格匹配模型输入shape）
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        # 1. 转换为numpy数组，shape=(1, 28, 28, 1)，完全匹配CNN输入
        image = np.array(data['image'], dtype=np.float32).reshape(1, 28, 28, 1)
        
        # 2. 模型预测，verbose=0关闭日志
        pred = model.predict(image, verbose=0)
        digit = int(np.argmax(pred))
        confidence = float(np.max(pred))
        
        return jsonify({
            "digit": digit,
            "confidence": confidence
        })
    except Exception as e:
        print(f"预测错误: {e}")
        return jsonify({"digit": -1, "confidence": 0})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)