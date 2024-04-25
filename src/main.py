from fastapi import FastAPI,File,UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import argparse
import uvicorn
import cv2
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import json
from json import JSONEncoder
import base64
import os
import cloudinary.uploader

# Initializing a FastAPI
app = FastAPI()

# Adding CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "PUT", "DELETE", "GET", "OPTIONS"],
)


# Configuring cloudinary with credentials
cloudinary.config(
  cloud_name="harshitcloud",
  api_key="421415441673459",
  api_secret='8iqlOAjBj3XKrgNQl0W9Oc6o_NU'
)

# Endpoint for detecting faces and making bounding boxes
@app.put("/ml/make_bounding_box")
async def make_bounding_box(image: UploadFile = File(...)):
    # Reading image data
    image_data = await image.read()
    
    # Decoding image
    decoded = cv2.imdecode(np.frombuffer(image_data, np.uint8), -1)
    
    # Converting image to grayscale
    gray_scale = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
    
    # Load img face classifier
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    # Flag to indicate if face is present
    face_present = False
    
    # Detecting faces in the image
    face = face_classifier.detectMultiScale(
        gray_scale, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    # Drawing bounding boxes around detected faces
    for (x, y, w, h) in face:
        cv2.rectangle(decoded, (x, y), (x + w, y + h), (0, 255, 0), 4)
        face_present = True

    # Converting image to RGB for display
    img_rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

    img = Image.fromarray(img_rgb)
    
    # Displaying image
    img.show()
    
    # Converting image to base64 string
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # Saving the image to a temporary file
    with open('temp.jpg', 'wb') as f:
        f.write(buffered.getvalue())

    # Uploading the image to Cloudinary
    upload_result = cloudinary.uploader.upload('temp.jpg')

    # Deleting the temporary file
    os.remove('temp.jpg')

    result = {
        "face_present": face_present,
        "data": upload_result['secure_url']  # The URL of the uploaded image
    }
    
    # Returning response
    return JSONResponse(content=jsonable_encoder({"status": 200, "message": "annotated", "result": result}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FastApi app")
    parser.add_argument("-p", "--port", type=int, default=8084)
    args = parser.parse_args()
    uvicorn.run("main:app", host="127.0.0.1", port=args.port, reload=True)