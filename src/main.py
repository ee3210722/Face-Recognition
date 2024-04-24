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

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST","PUT","DELETE","GET","OPTIONS"],
)


# Configure cloudinary with your credentials
cloudinary.config(
  cloud_name = "harshitcloud",
  api_key =  "421415441673459",
  api_secret = '8iqlOAjBj3XKrgNQl0W9Oc6o_NU'
)

@app.put("/ml/make_bounding_box")
async def make_bounding_box(image: UploadFile = File(...)):
    image_data=await image.read()
    decoded=cv2.imdecode(np.frombuffer(image_data, np.uint8),-1)
    gray_scale=cv2.cvtColor(decoded,cv2.COLOR_BGR2GRAY)
    face_classifier=cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    face_present=False
    face = face_classifier.detectMultiScale(
        gray_scale, scaleFactor=1.1, minNeighbors=5, minSize=(40,40)
    )

    for (x,y,w,h) in face:
        cv2.rectangle(decoded,(x,y),(x+w,y+h),(0,255,0),4)
        face_present=True
    img_rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    img=Image.fromarray(img_rgb)
    img.show()
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # Save the image to a temporary file
    with open('temp.jpg', 'wb') as f:
        f.write(buffered.getvalue())

    # Upload the image to Cloudinary
    upload_result = cloudinary.uploader.upload('temp.jpg')

    # Delete the temporary file
    os.remove('temp.jpg')

    result={
        "face_present":face_present,
        "data": upload_result['secure_url']  # The URL of the uploaded image
    }
    return JSONResponse(content=jsonable_encoder({"status":200,"message":"annotated","result":result}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Run FastApi app")
    parser.add_argument("-p","--port",type=int,default=8084)
    args=parser.parse_args()
    uvicorn.run("main:app",host="127.0.0.1",port=args.port,reload=True)