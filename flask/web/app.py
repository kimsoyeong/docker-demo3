import time
# import pymysql

from flask import Flask, request, jsonify, send_file
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io

# Load model
base_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

placeholder = tf.keras.layers.Input(shape=(None, None, 3), dtype=tf.float32)
placeholder_1 = tf.keras.layers.Input(shape=(None, None, 3), dtype=tf.float32)
net = hub.KerasLayer(base_model, signature='serving_default', signature_outputs_as_dict=True)({'placeholder': placeholder, 'placeholder_1': placeholder_1})
model = tf.keras.models.Model({'placeholder': placeholder, 'placeholder_1': placeholder_1}, net)


# predict
app = Flask(__name__)

@app.route('/')
def hello():
    return '<h1>HELLO</h1>'

@app.route('/send_image', methods=['POST', 'GET'])
def send_image():
    if request.method == "POST":
        if request.files.get("myImage") and request.files.get("styleImage") :
            image = request.files["myImage"].read()
            image = Image.open(io.BytesIO(image)).convert('RGB')
            # image.show()
            width, height = image.size
            image_numpy = np.array(image)
            x_test = np.array([image_numpy])
            x_test = x_test / 255
            content_image = x_test

            image = request.files["styleImage"].read()
            image = Image.open(io.BytesIO(image)).convert('RGB')
            # image.show()
            image = image.resize((width, height))
            image_numpy = np.array(image)
            x_test = np.array([image_numpy])
            x_test = x_test / 255
            style_image = x_test

            dict_outut = model.predict({'placeholder': content_image, 'placeholder_1': style_image})

            y_predict = dict_outut['output_0']
            numpy_image = y_predict[0]
            numpy_image = (numpy_image * 255).astype(np.uint8)
            image = Image.fromarray(numpy_image)
            bytesIO = io.BytesIO()
            image.save(bytesIO, 'JPEG', quality=70)
            bytesIO.seek(0)
            image.show()
            
            return send_file(bytesIO, mimetype='image/jpeg')
            # return jsonify({"result": "test mode"})
        else:
            return jsonify({"result": "failed to get image"})
    else:
        return "not post mode"


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

