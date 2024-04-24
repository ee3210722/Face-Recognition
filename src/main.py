from fastapi import FastAPI,File,UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
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

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

app=FastAPI()

# def check_bounding_box():



@app.get("/make_bounding_box")
async def make_bounding_box(image: UploadFile = File(...)):
    image_data=await image.read()
    decoded=cv2.imdecode(np.frombuffer(image_data, np.uint8),-1)
    gray_scale=cv2.cvtColor(decoded,cv2.COLOR_BGR2GRAY)
    face_classifier=cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    face = face_classifier.detectMultiScale(
        gray_scale, scaleFactor=1.1, minNeighbors=5, minSize=(40,40)
    )
    for (x,y,w,h) in face:
        cv2.rectangle(decoded,(x,y),(x+w,y+h),(0,255,0),4)
    img_rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    img=Image.fromarray(img_rgb)
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return JSONResponse(content=jsonable_encoder({"status":200,"message":"annotated","result":img_str}))
if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Run FastApi app")
    parser.add_argument("-p","--port",type=int,default=8084)
    args=parser.parse_args()
    uvicorn.run("main:app",host="127.0.0.1",port=args.port,reload=True)