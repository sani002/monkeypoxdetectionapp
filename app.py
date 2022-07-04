import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

def get_model():
    global model
    model = load_model('model.h5')

def load_image(img_path):

    img = image.load_img(img_path, target_size=(256, 256))
    img_tensor = image.img_to_array(img)                   
    img_tensor = np.expand_dims(img_tensor, axis=0)       
    img_tensor /= 255.                                     

    return img_tensor

def prediction(img_path):
    new_image = load_image(img_path)
    
    pred = model.predict(new_image)
    
    print(pred)
    if pred<0.5:
        return "It might be Monkeypox. You should visit a specialist immediately. Thank you."
    else:
        return "It's most probably not monkeypox, but still you should visit a skin specialist. Thank you."

get_model()

@app.route("/", methods=['GET', 'POST'])
def home():

    return render_template('home.html')

@app.route("/predict", methods = ['GET','POST'])
def predict():
    
    if request.method == 'POST':
        
        file = request.files['file']
        filename = file.filename
        file_path = os.path.join(r'/static', filename)    #for web deployment remember to change static location!                  
        file.save(file_path)
        print(filename)
        product = prediction(file_path)
        print(product)
        
    return render_template('predict.html', product = product, user_image = file_path)  

if __name__ == "__main__":
    app.run(debug=False,host='0.0.0.0')
