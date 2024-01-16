import os
import ssl
import urllib.request

import cv2
import numpy as np
from ast import literal_eval

from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageFilter

from app import cartoonize
from app import guided_filter
from app import network

app = FastAPI()

# SSL configuration for HTTPS requests
ssl._create_default_https_context = ssl._create_unverified_context

# CORS configuration: specify the origins that are allowed to make cross-site requests
origins = [
    "https://localhost:8080",
    "https://localhost:8080/",
    "http://localhost:8080",
    "http://localhost:8080/",
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# A simple endpoint to verify that the API is online.
@app.get("/")
def home():
    return {"Test": "Online"}


@app.get("/get-cartoon/{cldId}/{imgId}/{sliderValues}")
async def get_cartoon(cldId: str, imgId: str,sliderValues: str, background_tasks: BackgroundTasks):    
    
    img_path = f"app/bib/{imgId}.jpg"
    image_url = f"https://cmp.photoprintit.com/api/photos/{imgId}.org?size=original&errorImage=false&cldId={cldId}&clientVersion=0.0.1-medienVerDemo"
    img_path_final = f"app/bib/temp.jpg"
    
    #values = np.array(literal_eval(sliderValues))

    download_image(image_url, img_path)
    apply_cartoon(img_path, sliderValues, img_path_final)

    # Schedule the image file to be deleted after the response is sent
    background_tasks.add_task(remove_file, img_path)

    # Send the blurred image file as a response
    return FileResponse(img_path_final)

@app.get("/cartoon-ai/{cldId}/{imgId}")
async def cartoonAI(cldId: str, imgId: str, background_tasks: BackgroundTasks):
    img_path = f"app/bib/{imgId}.jpg"
    image_url = f"https://cmp.photoprintit.com/api/photos/{imgId}.org?size=original&errorImage=false&cldId={cldId}&clientVersion=0.0.1-medienVerDemo"
    img_path_final = f"app/bib/temp.jpg"
    
    download_image(image_url, img_path)
    apply_ai(img_path,img_path_final)
    
    # Schedule the image file to be deleted after the response is sent
    background_tasks.add_task(remove_file, img_path)

    # Send the blurred image file as a response
    return FileResponse(img_path_final)

# Downloads an image from the specified URL and saves it to the given path.
def download_image(image_url: str, img_path: str):
    urllib.request.urlretrieve(image_url, img_path)


# Opens the image from the given path and applies a box blur effect.
def apply_cartoon(img_path: str, sliderValues, img_path_final):
    #load values and convert string to numpy array
    values = np.array(literal_eval(sliderValues))
    #load image
    imgbase = cv2.imread(img_path)
    imgbase = cv2.cvtColor(imgbase, cv2.COLOR_BGR2RGB)
    #contrast
    lab= cv2.cvtColor(imgbase, cv2.COLOR_RGB2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(values[0],values[0]))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl,a,b))
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    #grayscale
    grayimg = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    grayimg = cv2.medianBlur(grayimg, 5)
    #edge filter
    edges = cv2.adaptiveThreshold(grayimg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, values[1], values[1])
    #color quantization
    data = np.float32(img).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    ret, label, center = cv2.kmeans(data, values[2], None, criteria, values[3], cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    #final image
    color = cv2.bilateralFilter(result, values[4], 250, 250) #can change the kernel size for this filter
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    final = Image.fromarray(cartoon)
    final.save(img_path_final)
    
def apply_ai(img_path: str, img_path_final):
    cartoonize.cartoonize(img_path, img_path_final, 'app/saved_models')
    

# Deletes the file at the specified path.
def remove_file(path: str):
    os.unlink(path)


# Global exception handler that catches all exceptions not handled by specific exception handlers.
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Print the exception type and its message for debugging
    print(f"Exception type: {type(exc).__name__}")
    print(f"Exception message: {str(exc)}")

    return JSONResponse(
        status_code=500,
        content={"message": "An unexpected error occurred."},
    )
