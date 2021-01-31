# 1. Library imports
import uvicorn
from fastapi import FastAPI, File, UploadFile
from prediction import read_image, predict, detect_face
import time
import cv2
import numpy as np 

# 2. Create the app object
app = FastAPI()


@app.get("/")
def index():
    """
    This is a first docstring.
    """
    return {"message": "Hello, WORLD"}


@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    """
    Predict gender of people in picture.
    """
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    start = time.time()

    image = read_image(await file.read())
    face, normalize = detect_face(image)
    normalize_face = face
    if face is not None:
        if normalize:
            normalize_face = np.array(face*255.0, dtype=np.int)
        # cv2.imwrite('Face.jpg', normalize_face)
        gender, emotion, cap = predict(normalize_face)
        end = time.time()
        print(f"Total prediction time : {end-start:.2f} seconds.")
        return gender, emotion, cap
    return "No Face detected..."


# 5. Run the API with uvicorn
if __name__ == "__main__":
    uvicorn.run(app)
