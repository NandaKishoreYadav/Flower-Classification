from flask import *
from keras.models import model_from_json
from keras.models import load_model
import numpy as np
from keras.utils import load_img, img_to_array
import os

json_file = open(r"Saved models\99 acc\model_flwr.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(r"Saved models\99 acc\model1_flwr.h5")

def flower(result):
    arr={'daisy': 0, 'dandelion': 1, 'rose': 2, 'sunflower': 3, 'tulip': 4}
    d={}
    for i in arr:
        d[arr[i]]=i
    maxi=0
    ans=''
    for i in range(len(result[0])):
        if maxi<result[0][i]:
            ans = d[i]
            maxi=result[0][i]
    return ans

def prediction(img_link):
    test_image=load_img(img_link,target_size=(200,200))
    test_image=img_to_array(test_image)
    test_image=np.expand_dims(test_image,axis=0)
    result=model.predict(test_image)
    return flower(result)

app=Flask(__name__)

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/predict',methods=["GET","POST"])
def predict():
    file=request.files['file']
    file_path = 'static/Storage/' + file.filename
    file.save(file_path)  
    k=prediction(file_path)
    k=k.capitalize()
    return render_template('prediction.html',ans=k,file='Storage/' + file.filename)

if __name__ == '__main__':
    app.run(debug=True)