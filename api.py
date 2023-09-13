from PIL import Image
import numpy as np
from fastapi import FastAPI, UploadFile, File
from io import BytesIO
import tensorflow as tf
import keras 
import os 
import cv2

app = FastAPI()

class_names = ['cataract', 'normal']
model = keras.models.load_model("catnet.h5")

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image

def predict(img_path):  
    img = tf.keras.utils.load_img(img_path, target_size=(256, 256))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    score = predictions[0]
    res = {
        "diagnosis":"{}".format(class_names[np.argmax(score)], 100 * np.max(score)),
        "confidence":"{:.4f}".format(100 * np.max(score)),
        }
    return res

def thresh_crop_image(img) :
    path = os.path.abspath(img)
    name = os.path.basename(img)
    img = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    hh, ww = thresh.shape
    thresh[hh-3:hh, 0:ww] = 0
    white = np.where(thresh==255)
    xmin, ymin, xmax, ymax = np.min(white[1]), np.min(white[0]), np.max(white[1]), np.max(white[0])
    crop = img[ymin:ymax+3, xmin:xmax]
    cv2.imwrite(path, crop)
    return path


IMAGEDIR = "images/"
if not os.path.exists(IMAGEDIR) :
    os.mkdir(IMAGEDIR)

@app.get('/')
async def root():
    return {"Response":"Wokring Great!"}

@app.post("/predict/")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg, jpeg, or png format!"

    contents = await file.read()
    with open(f"{IMAGEDIR}{file.filename}", "wb") as f:
        f.write(contents)

    file_path = f"{IMAGEDIR}{file.filename}"
    threshed = thresh_crop_image(file_path)
    prediction = predict(threshed)
    os.remove(file_path)
    return prediction