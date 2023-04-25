{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/emirkocer/ms-api/blob/main/app.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "RyCwRS9qUOQs"
      },
      "outputs": [],
      "source": [
        "!pip3 install uvicorn\n",
        "!pip3 install fastapi\n",
        "!pip3 install python-multipart\n",
        "!pip3 install nest-asyncio\n",
        "!pip3 install pyngrok\n",
        "!pip3 install requests\n",
        "!pip3 install scikit-learn"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "FYop3ivKCg4d",
        "outputId": "bf418a97-1dec-4dff-d569-1440c7ce8069"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Authtoken saved to configuration file: /root/.ngrok2/ngrok.yml\n"
          ]
        }
      ],
      "source": [
        "!ngrok config add-authtoken 2OHYxeeihdbNVqb80Wh5fI9YDj9_o8ka2omKmLQWUZCTVNNA"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "HFJ1Z3_Qfcdz"
      },
      "outputs": [],
      "source": [
        "import io\n",
        "import sys\n",
        "import logging\n",
        "import nest_asyncio\n",
        "import uvicorn\n",
        "from typing import IO, List\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "from pyngrok import ngrok\n",
        "from fastapi import FastAPI, File, UploadFile\n",
        "from fastapi.responses import StreamingResponse\n",
        "from fastapi.middleware.cors import CORSMiddleware\n",
        "import cv2\n",
        "import zipfile\n",
        "from PIL import Image\n",
        "import torch\n",
        "import tensorflow as tf\n",
        "import tensorflow.keras.backend as K\n",
        "from google.colab import drive\n",
        "from sklearn.metrics import f1_score"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "FvWlPx4Gt5Au"
      },
      "outputs": [],
      "source": [
        "# Logging\n",
        "logging.basicConfig(level=logging.DEBUG, \n",
        "                    format='%(asctime)s %(levelname)s %(message)s',\n",
        "                    force=True,\n",
        "                    )"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "DZ4QxKdRffY4"
      },
      "outputs": [],
      "source": [
        "# API app\n",
        "app = FastAPI(\n",
        "    title=\"Brain MRI Segmentation API\",\n",
        "    description=\"\"\"This is a REST API that takes a brain MRI \n",
        "    image as input and returns its segmented version using a \n",
        "    pre-trained U-Net model.\"\"\",\n",
        "    version=\"0.1\",\n",
        "    )"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "lrQ4iDdAgPth"
      },
      "outputs": [],
      "source": [
        "#CORS\n",
        "origins = [\n",
        "    \"http://localhost\",\n",
        "    \"http://localhost:8000\",\n",
        "    \"https://localhost\",\n",
        "    \"https://localhost:8000\",\n",
        "    \"*\",\n",
        "]\n",
        "\n",
        "app.add_middleware(\n",
        "    CORSMiddleware,\n",
        "    allow_origins=origins,\n",
        "    allow_credentials=True,\n",
        "    allow_methods=[\"*\"],\n",
        "    allow_headers=[\"*\"],\n",
        ")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "ZqPSAPC2M8E1"
      },
      "outputs": [],
      "source": [
        "# Unzip pretrained model zip file\n",
        "def unzip_model(file_path, destination_path):\n",
        "    with zipfile.ZipFile(file_path, 'r') as zip_ref:\n",
        "        zip_ref.extractall(destination_path)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "MEJk9dRmU3kR"
      },
      "outputs": [],
      "source": [
        "# Dice coefficient\n",
        "def dice_coef(y_true, y_pred, smooth=1):\n",
        "    y_true_f = K.flatten(y_true)\n",
        "    y_pred_f = K.flatten(y_pred)\n",
        "    intersection = K.sum(y_true_f * y_pred_f)\n",
        "    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "GqviNX28VvGM"
      },
      "outputs": [],
      "source": [
        "# Jaccard score\n",
        "def iou_coef(y_true, y_pred, smooth=1):\n",
        "    y_true_f = tf.cast(y_true, dtype='float32')\n",
        "    y_pred_f = tf.cast(y_pred, dtype='float32')\n",
        "    y_true_f = tf.expand_dims(y_true_f, axis=0)\n",
        "    y_pred_f = tf.expand_dims(y_pred_f, axis=0)\n",
        "    intersection = tf.reduce_sum(tf.math.abs(y_true_f * y_pred_f), axis=[1, 2])\n",
        "    union = tf.reduce_sum(y_true_f, axis=[1, 2]) + tf.reduce_sum(y_pred_f, axis=[1, 2]) - intersection\n",
        "    iou = tf.reduce_mean((intersection + tf.keras.backend.epsilon()) / (union + tf.keras.backend.epsilon()), axis=0)\n",
        "    return iou"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "-hoK-mTAWiUg"
      },
      "outputs": [],
      "source": [
        "# Dice loss (required to load the model)\n",
        "def dice_loss(y_true, y_pred):\n",
        "    return 1 - dice_coef(y_true, y_pred)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "YbY8cdaGtNnm"
      },
      "outputs": [],
      "source": [
        "# Pre-trained U-net model (import from Drive)\n",
        "model = None\n",
        "def load_model():\n",
        "    global model\n",
        "    try:\n",
        "        logging.info(\"Loading the model...\")\n",
        "\n",
        "        zip_file_path = \"/content/drive/MyDrive/NLP/Models/brain_segmentation_trained_model.zip\"\n",
        "        model_directory = \"/content/brain_mri_model/trained_model/trained_model\"\n",
        "\n",
        "        unzip_model(zip_file_path, model_directory)\n",
        "\n",
        "        #check \n",
        "        custom_objects = {\n",
        "            'dice_coef': dice_coef,\n",
        "            'dice_loss': dice_loss,\n",
        "            'iou_coef': iou_coef\n",
        "        }\n",
        "\n",
        "        model = tf.keras.models.load_model(model_directory, custom_objects=custom_objects)\n",
        "\n",
        "        logging.info(\"Model loaded successfully.\")\n",
        "    except Exception as e:\n",
        "        logging.error(f\"Error loading model: {e}\")\n",
        "        model = None"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "umlb9bN5qFCi"
      },
      "outputs": [],
      "source": [
        "# Open the image\n",
        "async def open_image(file: IO[bytes]):\n",
        "    try:\n",
        "        image_bytes = await file.read()\n",
        "        logging.info(f\"Opening image with size {len(image_bytes)} bytes\")\n",
        "        img = Image.open(io.BytesIO(image_bytes))\n",
        "        logging.info(f\"Image opened successfully with size {img.size}\")\n",
        "        return img, image_bytes\n",
        "    except Exception as e:\n",
        "        logging.error(f\"Error opening image: {e}\")\n",
        "        return None"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "fG7zJpaBchdR"
      },
      "outputs": [],
      "source": [
        "# Check the image format\n",
        "async def check_image_format(file: bytes) -> str:\n",
        "    try:\n",
        "        image_bytes = await file.read()\n",
        "        img = Image.open(io.BytesIO(image_bytes))\n",
        "        img.verify()\n",
        "        return img.format.lower()\n",
        "    except Exception as e:\n",
        "        logging.error(f\"Invalid image format: {e}\")\n",
        "        raise ValueError(\"Invalid image format.\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "MPtfOuZMqWVx"
      },
      "outputs": [],
      "source": [
        "# Preprocessing\n",
        "def transform_image(img):\n",
        "    try:\n",
        "        logging.info(\"Transforming image.\")\n",
        "\n",
        "        img_resized = img.resize((256, 256))\n",
        "        img_np = np.array(img_resized) / 255.0\n",
        "        img_batch = np.expand_dims(img_np, axis=(0, -1))\n",
        "\n",
        "        logging.info(\"Image transformed successfully.\")\n",
        "\n",
        "        return img_batch\n",
        "    except Exception as e:\n",
        "        logging.error(f\"Error during image transformation: {e}\")\n",
        "        return None\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "VpIKr9XpqaoA"
      },
      "outputs": [],
      "source": [
        "# Prediction \n",
        "def predict(image_tensor, original_image):\n",
        "    try:\n",
        "        logging.info(\"Prediction started...\")\n",
        "        \n",
        "        output = model.predict(image_tensor)\n",
        "        output = output.squeeze()\n",
        "\n",
        "        logging.info(\"Prediction completed. Creating binary mask...\")\n",
        "\n",
        "        # Mask\n",
        "        threshold = 0.5\n",
        "        binary_mask = (output > threshold).astype('uint8')\n",
        "\n",
        "        plt.imshow(binary_mask, cmap='gray')\n",
        "        plt.show() \n",
        "\n",
        "        logging.info(\"Segmented image is predicted successfully!\")\n",
        "\n",
        "        return binary_mask\n",
        "    except Exception as e:\n",
        "        logging.error(f\"Error during prediction: {e}\")\n",
        "        return None"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "elTXxJRydeEo"
      },
      "outputs": [],
      "source": [
        "# Bytes to iterator\n",
        "def iter_bytes(data: bytes, chunk_size: int = 8192):\n",
        "    start = 0\n",
        "    end = chunk_size\n",
        "    while start < len(data):\n",
        "        yield data[start:end]\n",
        "        start += chunk_size\n",
        "        end += chunk_size"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Dz6gL1vqQyUm"
      },
      "outputs": [],
      "source": [
        "# Optimal batch size\n",
        "def get_batch_size():\n",
        "    if torch.cuda.is_available():\n",
        "        total_memory = torch.cuda.get_device_properties(0).total_memory\n",
        "        free_memory = torch.cuda.memory_reserved(0)\n",
        "        available_memory = total_memory - free_memory\n",
        "\n",
        "        if available_memory > (10 * (1024 ** 3)):  #check this\n",
        "            batch_size = 32\n",
        "        elif available_memory > (6 * (1024 ** 3)):  \n",
        "            batch_size = 16\n",
        "        else:\n",
        "            batch_size = 8\n",
        "    else:\n",
        "        batch_size = 4  \n",
        "        \n",
        "    return batch_size"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "mVn2efR3SGHX"
      },
      "outputs": [],
      "source": [
        "# Create batches\n",
        "def create_batches(images, masks, batch_size):\n",
        "    image_batches = [images[i:i + batch_size] \n",
        "                     for i in range(0, len(images), batch_size)]\n",
        "    mask_batches = [masks[i:i + batch_size] \n",
        "                    for i in range(0, len(masks), batch_size)]\n",
        "    return list(zip(image_batches, mask_batches))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "LqpcVv1ddfLp"
      },
      "outputs": [],
      "source": [
        "# Ngrok tunnel\n",
        "def setup_ngrok_tunnel(port_id):\n",
        "  return ngrok.connect(str(port_id))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "8LodzXkShK0n"
      },
      "outputs": [],
      "source": [
        "# Uvicorn\n",
        "def run_app(port_id: int, app: FastAPI):\n",
        "  uvicorn.run(app, port=port_id)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "rf0N3745owSN"
      },
      "outputs": [],
      "source": [
        "# Root\n",
        "@app.get(\"/\")\n",
        "def read_root():\n",
        "    return {\"message\": \"Welcome to the Brain MRI Segmentation API!\"}"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "-v8k1w4ZqjQO"
      },
      "outputs": [],
      "source": [
        "# Predict route \n",
        "@app.post(\"/predict\", summary=\"Single Image Prediction\",\n",
        "          description=\"\"\"This endpoint accepts a single\n",
        "          brain MRI image and returns its segmented version\n",
        "          using a pre-trained U-Net model.\"\"\"\n",
        "          )\n",
        "async def predict_handler(file: UploadFile = File(...)):\n",
        "\n",
        "    logging.info(\"Setting up the prediction...\")\n",
        "\n",
        "    try:\n",
        "        logging.info(\"Received image for prediction.\")\n",
        "\n",
        "        image_format = await check_image_format(file)\n",
        "\n",
        "        logging.info(f\"Supported image format: {image_format}\")\n",
        "\n",
        "        file.file.seek(0)\n",
        "\n",
        "        image_original, image_bytes = await open_image(file)\n",
        "\n",
        "        if image_original is not None:\n",
        "            image_tensor = transform_image(image_original)\n",
        "        else:\n",
        "            return {\"message\": \"Error during image opening.\"}\n",
        "\n",
        "        if image_tensor is not None:\n",
        "            prediction = predict(image_tensor, image_original)\n",
        "        else:\n",
        "            return {\"message\": \"Error during image transformation.\"}\n",
        "\n",
        "        if prediction is not None:\n",
        "            prediction_img = Image.fromarray(prediction * 255).convert(\"L\")\n",
        "            buffer = io.BytesIO()\n",
        "            prediction_img.save(buffer, format=\"PNG\")\n",
        "            return StreamingResponse(iter_bytes(buffer.getvalue()), media_type=\"image/png\")\n",
        "        else:\n",
        "            return {\"message\": \"Error during prediction. Please try again later.\"}\n",
        "    except Exception as e:\n",
        "        logging.error(f\"Error during image upload: {e}\")\n",
        "        return {\"message\": \"Error during image upload. Please try again later.\"}\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "p2DM_XHY6V6V"
      },
      "outputs": [],
      "source": [
        "# Batch prediction route\n",
        "@app.post(\"/batch_predict\", summary=\"Batch Segmentation and Metrics\",\n",
        "          description=\"\"\"This endpoint accepts a batch of brain MRI \n",
        "          images and their corresponding ground truth masks, \n",
        "          performs segmentation on each image, and returns the \n",
        "          Jaccard score and F1 score as metrics.\"\"\"\n",
        "          )\n",
        "async def batch_predict_handler(images: List[UploadFile] = File(...), \n",
        "                                masks: List[UploadFile] = File(...)):\n",
        "    logging.info(\"Received images for prediction.\")\n",
        "\n",
        "    batch_size = get_batch_size()\n",
        "\n",
        "    logging.info(f\"Batch size is determined as {batch_size}\")\n",
        "\n",
        "    batches = create_batches(images, masks, batch_size)\n",
        "\n",
        "    logging.info(\"Batches are successfully created.\")\n",
        "\n",
        "    if len(images) != len(masks):\n",
        "        return {\"message\": \"The number of images and masks should be the same.\"}\n",
        "    \n",
        "    results = []\n",
        "    for image_batch, mask_batch in batches:\n",
        "        batch_results = []\n",
        "        for image, mask in zip(image_batch, mask_batch):\n",
        "\n",
        "            img_format = await check_image_format(image)\n",
        "            mask_format = await check_image_format(mask)\n",
        "\n",
        "            logging.info(f\"Supported image format: {img_format}\")\n",
        "            logging.info(f\"Supported mask format: {mask_format}\")\n",
        "\n",
        "            image.file.seek(0)\n",
        "            mask.file.seek(0)\n",
        "\n",
        "            image_original, _ = await open_image(image)\n",
        "            image_mask, _ = await open_image(mask)\n",
        "\n",
        "            if image_original is not None:\n",
        "                image_tensor = transform_image(image_original)\n",
        "            else:\n",
        "                return {\"message\": \"Error during image transformation.\"}\n",
        "\n",
        "            if image_tensor is not None:\n",
        "                binary_mask = predict(image_tensor, image_original)\n",
        "            else:\n",
        "                return {\"message\": \"Error during binary mask prediction.\"}\n",
        "\n",
        "            if binary_mask is not None:\n",
        "                logging.info(\"Metrics are being calculated...\")\n",
        "                image_mask = np.array(image_mask.convert(\"L\"))\n",
        "                image_mask = (image_mask > 127).astype('uint8')\n",
        "\n",
        "                # jaccard\n",
        "                iou_score = iou_coef(binary_mask, image_mask).numpy().item()\n",
        "\n",
        "                # dice\n",
        "                if np.sum(binary_mask) + np.sum(image_mask) == 0:\n",
        "                    dice = 1.0\n",
        "                else:\n",
        "                    dice = f1_score(binary_mask.ravel(), image_mask.ravel(), average='binary')\n",
        "\n",
        "                batch_results.append({\"image_url\": image.filename, \n",
        "                                      \"metrics\": {\"jaccard\": iou_score, \"dice_coefficient\": dice}})\n",
        "            else:\n",
        "                return {\"message\": \"Error during prediction.\"}\n",
        "        \n",
        "        results.extend(batch_results)\n",
        "\n",
        "    return results"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "background_save": true
        },
        "id": "tfZ8pR2QvsMv"
      },
      "outputs": [],
      "source": [
        "if __name__ == \"__main__\":\n",
        "\n",
        "    load_model()\n",
        "\n",
        "    port = 8000\n",
        "    \n",
        "    nest_asyncio.apply() # to avoid runtime error\n",
        "\n",
        "    public_url = setup_ngrok_tunnel(port)\n",
        "    print(\"Public URL: \", public_url)\n",
        "\n",
        "    run_app(port, app) "
      ]
    }
  ],
  "metadata": {
    "colab": {
      "provenance": [],
      "mount_file_id": "1mFLWpVKkb5gn4U0Twc2K_Yf-22e-U0uI",
      "authorship_tag": "ABX9TyOn5OO55qpID3uLtsmUKtjK",
      "include_colab_link": true
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}