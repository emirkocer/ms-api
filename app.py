import io
import sys
import logging
import nest_asyncio
import uvicorn
from typing import IO, List
import numpy as np
import matplotlib.pyplot as plt
from pyngrok import ngrok
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import zipfile
from PIL import Image
import torch
import tensorflow as tf
import tensorflow.keras.backend as K
from google.colab import drive
from sklearn.metrics import f1_score

# Logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)s %(message)s',
                    force=True,
                    )

# API app
app = FastAPI(
    title="Brain MRI Segmentation API",
    description="""This is a REST API that takes a brain MRI 
    image as input and returns its segmented version using a 
    pre-trained U-Net model.""",
    version="0.1",
    )

#CORS
origins = [
    "http://localhost",
    "http://localhost:8000",
    "https://localhost",
    "https://localhost:8000",
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Unzip pretrained model zip file
def unzip_model(file_path, destination_path):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(destination_path)

# Dice coefficient
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# Jaccard score
def iou_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.cast(y_true, dtype='float32')
    y_pred_f = tf.cast(y_pred, dtype='float32')
    y_true_f = tf.expand_dims(y_true_f, axis=0)
    y_pred_f = tf.expand_dims(y_pred_f, axis=0)
    intersection = tf.reduce_sum(tf.math.abs(y_true_f * y_pred_f), axis=[1, 2])
    union = tf.reduce_sum(y_true_f, axis=[1, 2]) + tf.reduce_sum(y_pred_f, axis=[1, 2]) - intersection
    iou = tf.reduce_mean((intersection + tf.keras.backend.epsilon()) / (union + tf.keras.backend.epsilon()), axis=0)
    return iou

# Dice loss (required to load the model)
def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

# Pre-trained U-net model (import from Drive)
model = None
def load_model():
    global model
    try:
        logging.info("Loading the model...")

        zip_file_path = "/content/drive/MyDrive/NLP/Models/brain_segmentation_trained_model.zip"
        model_directory = "/content/brain_mri_model/trained_model/trained_model"

        unzip_model(zip_file_path, model_directory)

        #check 
        custom_objects = {
            'dice_coef': dice_coef,
            'dice_loss': dice_loss,
            'iou_coef': iou_coef
        }

        model = tf.keras.models.load_model(model_directory, custom_objects=custom_objects)

        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        model = None

# Open the image
async def open_image(file: IO[bytes]):
    try:
        image_bytes = await file.read()
        logging.info(f"Opening image with size {len(image_bytes)} bytes")
        img = Image.open(io.BytesIO(image_bytes))
        logging.info(f"Image opened successfully with size {img.size}")
        return img, image_bytes
    except Exception as e:
        logging.error(f"Error opening image: {e}")
        return None

# Check the image format
async def check_image_format(file: bytes) -> str:
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()
        return img.format.lower()
    except Exception as e:
        logging.error(f"Invalid image format: {e}")
        raise ValueError("Invalid image format.")

# Preprocessing
def transform_image(img):
    try:
        logging.info("Transforming image.")

        img_resized = img.resize((256, 256))
        img_np = np.array(img_resized) / 255.0
        img_batch = np.expand_dims(img_np, axis=(0, -1))

        logging.info("Image transformed successfully.")

        return img_batch
    except Exception as e:
        logging.error(f"Error during image transformation: {e}")
        return None

# Prediction 
def predict(image_tensor, original_image):
    try:
        logging.info("Prediction started...")
        
        output = model.predict(image_tensor)
        output = output.squeeze()

        logging.info("Prediction completed. Creating binary mask...")

        # Mask
        threshold = 0.5
        binary_mask = (output > threshold).astype('uint8')

        plt.imshow(binary_mask, cmap='gray')
        plt.show() 

        logging.info("Segmented image is predicted successfully!")

        return binary_mask
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return None

# Bytes to iterator
def iter_bytes(data: bytes, chunk_size: int = 8192):
    start = 0
    end = chunk_size
    while start < len(data):
        yield data[start:end]
        start += chunk_size
        end += chunk_size

# Optimal batch size
def get_batch_size():
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        free_memory = torch.cuda.memory_reserved(0)
        available_memory = total_memory - free_memory

        if available_memory > (10 * (1024 ** 3)):  #check this
            batch_size = 32
        elif available_memory > (6 * (1024 ** 3)):  
            batch_size = 16
        else:
            batch_size = 8
    else:
        batch_size = 4  
        
    return batch_size

# Create batches
def create_batches(images, masks, batch_size):
    image_batches = [images[i:i + batch_size] 
                     for i in range(0, len(images), batch_size)]
    mask_batches = [masks[i:i + batch_size] 
                    for i in range(0, len(masks), batch_size)]
    return list(zip(image_batches, mask_batches))

# Ngrok tunnel
def setup_ngrok_tunnel(port_id):
  return ngrok.connect(str(port_id))

# Uvicorn
def run_app(port_id: int, app: FastAPI):
  uvicorn.run(app, port=port_id)

# Root
@app.get("/")
def read_root():
    return {"message": "Welcome to the Brain MRI Segmentation API!"}

# Predict route 
@app.post("/predict", summary="Single Image Prediction",
          description="""This endpoint accepts a single
          brain MRI image and returns its segmented version
          using a pre-trained U-Net model."""
          )
async def predict_handler(file: UploadFile = File(...)):

    logging.info("Setting up the prediction...")

    try:
        logging.info("Received image for prediction.")

        image_format = await check_image_format(file)

        logging.info(f"Supported image format: {image_format}")

        file.file.seek(0)

        image_original, image_bytes = await open_image(file)

        if image_original is not None:
            image_tensor = transform_image(image_original)
        else:
            return {"message": "Error during image opening."}

        if image_tensor is not None:
            prediction = predict(image_tensor, image_original)
        else:
            return {"message": "Error during image transformation."}

        if prediction is not None:
            prediction_img = Image.fromarray(prediction * 255).convert("L")
            buffer = io.BytesIO()
            prediction_img.save(buffer, format="PNG")
            return StreamingResponse(iter_bytes(buffer.getvalue()), media_type="image/png")
        else:
            return {"message": "Error during prediction. Please try again later."}
    except Exception as e:
        logging.error(f"Error during image upload: {e}")
        return {"message": "Error during image upload. Please try again later."}

# Batch prediction route
@app.post("/batch_predict", summary="Batch Segmentation and Metrics",
          description="""This endpoint accepts a batch of brain MRI 
          images and their corresponding ground truth masks, 
          performs segmentation on each image, and returns the 
          Jaccard score and F1 score as metrics."""
          )
async def batch_predict_handler(images: List[UploadFile] = File(...), 
                                masks: List[UploadFile] = File(...)):
    logging.info("Received images for prediction.")

    batch_size = get_batch_size()

    logging.info(f"Batch size is determined as {batch_size}")

    batches = create_batches(images, masks, batch_size)

    logging.info("Batches are successfully created.")

    if len(images) != len(masks):
        return {"message": "The number of images and masks should be the same."}
    
    results = []
    for image_batch, mask_batch in batches:
        batch_results = []
        for image, mask in zip(image_batch, mask_batch):

            img_format = await check_image_format(image)
            mask_format = await check_image_format(mask)

            logging.info(f"Supported image format: {img_format}")
            logging.info(f"Supported mask format: {mask_format}")

            image.file.seek(0)
            mask.file.seek(0)

            image_original, _ = await open_image(image)
            image_mask, _ = await open_image(mask)

            if image_original is not None:
                image_tensor = transform_image(image_original)
            else:
                return {"message": "Error during image transformation."}

            if image_tensor is not None:
                binary_mask = predict(image_tensor, image_original)
            else:
                return {"message": "Error during binary mask prediction."}

            if binary_mask is not None:
                logging.info("Metrics are being calculated...")
                image_mask = np.array(image_mask.convert("L"))
                image_mask = (image_mask > 127).astype('uint8')

                # jaccard
                iou_score = iou_coef(binary_mask, image_mask).numpy().item()

                # dice
                if np.sum(binary_mask) + np.sum(image_mask) == 0:
                    dice = 1.0
                else:
                    dice = f1_score(binary_mask.ravel(), image_mask.ravel(), average='binary')

                batch_results.append({"image_url": image.filename, 
                                      "metrics": {"jaccard": iou_score, "dice_coefficient": dice}})
            else:
                return {"message": "Error during prediction."}
        
        results.extend(batch_results)

    return results

if __name__ == "__main__":

    load_model()

    port = 8000
    
    nest_asyncio.apply() # to avoid runtime error

    public_url = setup_ngrok_tunnel(port)
    print("Public URL: ", public_url)

    run_app(port, app)
