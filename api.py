import tensorflow as tf
import keras
import numpy as np 
import cv2

# Function the cut the image to only the eye fundus
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


class_names = ['cataract', 'normal']
model = keras.models.load_model("catnet.h5")

def predict(img_path):  
    img = tf.keras.utils.load_img(img_path, target_size=(256, 256))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    score = predictions[0]
    # score = 100 * np.max(score)
    res = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score))
    res = {"{}".format(class_names[np.argmax(score)], 100 * np.max(score)):"{:.4f}".format(100 * np.max(score))}
    return res


from fastapi import FastAPI, File, UploadFile
import uuid
import os 

app = FastAPI()

IMAGEDIR = "images/"
if not os.path.exists(IMAGEDIR) :
    os.mkdir(IMAGEDIR)


@app.get('/')
def root():
    return {"Response":"Working Great!"}


@app.post('/diagnose')
async def create_upload_file(file: UploadFile = File(...)):
 
    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()
 
    #save the file
    with open(f"{IMAGEDIR}{file.filename}", "wb") as f:
        f.write(contents)

    file_path = f"{IMAGEDIR}{file.filename}"

    thresed_and_cropped = thresh_crop_image(file_path)
    diagnosis = predict(thresed_and_cropped)
    os.remove(file_path)

    return diagnosis