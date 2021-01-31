from PIL import Image
from io import BytesIO
import numpy as np
from deepface import DeepFace
from tensorflow.keras.models import model_from_json
import pandas as pd 
from face_recognition import face_locations
import time 
import cv2

# INPUT_SHAPE_1 = (480, 480)
INPUT_SHAPE_2 = (224, 224)


def read_image(image_encoded):
    pil_image = Image.open(BytesIO(image_encoded))
    return pil_image

def load_model(model_path):
    json_file = open(model_path, 'r')
    model_json = json_file.read()
    model = model_from_json(model_json)
    return model 

def load_tf_model():
    gmodel = load_model('./deepface/models/gender_model.json')
    emodel = load_model('./deepface/models/emotion_model.json')
    gmodel.load_weights('./deepface/models/gender_weights.h5')
    emodel.load_weights('./deepface/models/emotion_weights.h5')
    print("Loaded model from disk")
    return gmodel, emodel

gmodel, emodel = load_tf_model()
df = pd.read_excel('./data/Caption.xlsx', sheet_name=[1, 2, 3, 4, 5, 6], header=None)

def detect_face(image: Image.Image):
    # image = image.resize(INPUT_SHAPE_1)
    arr_image = np.asarray(image)
    try:
        s = time.time()
        detected_face = DeepFace.detectFace(arr_image, detector_backend = 'mtcnn')
        e = time.time()
        print(f'Detection time by DeepFace : {e-s} seconds')
    except Exception as err:
        e = time.time()
        print(f'Detection time by DeepFace : {e-s} seconds - No Face')
        s = time.time()
        image = image.resize(INPUT_SHAPE_2)
        arr_image = np.asarray(image)
        faces = face_locations(arr_image, model='cnn')
        e = time.time()
        print(f'Detection time by Face recognition Library: {e-s} seconds')
        sorted_faces = sorted(faces, key=lambda x: (x[1]-x[3])*(x[2]-x[0]), reverse=True)
        if len(sorted_faces) > 0:
            top, right, bottom, left = sorted_faces[0]
            arr_face = arr_image[top:bottom, left:right].copy()
            arr_face = cv2.resize(arr_face, INPUT_SHAPE_2)
            return arr_face, False 
        else:
            return None, False
    return detected_face, True

def generate_caption(gender, emotion):
    # Generate a caption for the image
    if gender == 'WOMEN':
        if emotion == 'SMILE':
            caption = df[1]        
        elif emotion == 'NORMAL':
            caption = df[2]
        else:
            caption = df[5]
    else:
        if emotion == 'SMILE':
            caption = df[3]
        elif emotion == 'NORMAL':
            caption = df[4]
        else:
            caption = df[6]

    num = np.random.randint(len(caption))
    return caption.iloc[num][0]
    
def predict(image:np.ndarray):
    image = np.expand_dims(image, 0)
    pred = gmodel.predict(image)
    print(pred)
    if pred[0] > 0.5:
        gender = 'WOMEN'
    else:
        gender = 'MEN'

    pred = emodel.predict(image)[0]
    if np.argmax(pred) == 0:
        emotion = 'SMILE'
    elif np.argmax(pred) == 1:
        emotion = 'NORMAL'
    else:
        emotion = 'WEAR_GLASSES'

    cap = generate_caption(gender, emotion)
    return gender, emotion, cap
