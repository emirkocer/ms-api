# FastAPI Brain MRI Segmentation

This repository contains a FastAPI implementation of a brain MRI segmentation service. The service accepts brain MRI images and returns segmented versions of the images using a pre-trained U-Net model.

This was a task I had to do for an interview process. I trained a U-net model and imported it from my own drive. You can use the structure of this API for different machine learning purposes. Ngrok tunnel was used for quick testing of the API, you need to sign up and authorize ngrok on your local machine to be able to run the API as it is.

Even if the code is not groundbreaking, I thought it can serve as a learning resource for others who are just starting with FastAPI or looking for ideas to implement similar projects.

## Overview

The main components of this project are:

- FastAPI: A modern, fast (high-performance) web framework for building APIs with Python 3.7+ based on standard Python type hints.
- U-Net: A convolutional neural network architecture for fast and precise segmentation of images.
- Swagger UI: An interactive interface for exploring and testing API endpoints.

## Usage

You can test the API using the Swagger UI ('predict' endpoint for single mask predictions) or the `curl` command ('batch_predict' endpoint for multiple predictions and returning metrics):

- Swagger UI: Open the public URL provided by ngrok in your web browser and append `/docs` to the URL.
- `curl`: Send a POST request with a file using `curl` (e.g., `curl -X POST -H "Content-Type: multipart/form-data" -F "file=@your_image_file.jpg" http://your_ngrok_url/predict`).

## License

This project is licensed under the terms of the MIT License. See the [LICENSE](LICENSE) file for details.
