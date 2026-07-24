# Bangladeshi Taka Note Detection (REST API + Docker)

This project detects the denomination of a Bangladeshi Taka note from an image. It uses a **YOLOv11** model for detection and serves it with a **FastAPI** REST API. The whole app is packed inside Docker so it can run anywhere.


When you send an image to the API, it tells you which note it is (like 100 taka), how confident the model is, and the bounding box of the note in the image.


**What it can detect**

The model knows 8 classes: **2 taka, 5 taka, 10 taka, 20 taka, 50 taka, 100 taka, 500 taka, 1000 taka**.


**Model accuracy**

The model was trained on a Kaggle T4 GPU using YOLOv11 (nano version). On the validation set of 600 images it got Precision 0.999, Recall 1.000, mAP@50 0.995 and mAP@50-95 0.995. The accuracy is very high because every image has one clear note filling the frame, so the detection task is easy for the model.


**Folder structure**

- app/ - the application code
- app/inference.py - loads the model and runs prediction
- app/main.py - FastAPI app with the /predict endpoint
- model/best.pt - trained YOLOv11 weights
- test_images/ - sample images to test the API
- outputs/ - saved prediction images
- notebooks/ - training notebook
- Dockerfile - docker build file
- requirements.txt - python dependencies
- README.md - this file
  

**About the dataset**

Dataset link: https://www.kaggle.com/datasets/rahnumatasnim1604103/bangladeshi-banknote-dataset


The dataset has around 70,000 images. The file name of each image tells the denomination. For example "100 (2571).png" means it is a 100 taka note.


The dataset did not come with bounding box labels. Since each image is just one note filling almost the whole frame, I made the YOLO labels automatically by putting a box that covers almost the full image. Then I took a balanced subset (300 train and 75 val per class) and trained YOLOv11 for 20 epochs.


**How to build the Docker image**

Make sure Docker Desktop is installed and running. Then from the project folder where the Dockerfile is, run this command:


 
`docker build -t taka-note-api .`

This installs PyTorch (CPU), Ultralytics and FastAPI, and copies the model and code into the image. The first build takes a few minutes.


**How to run the container**

Run this command:

 
`docker run -d -p 8000:8000 --name taka-api taka-note-api`

Check if it is running:


 
`docker ps`

`docker logs taka-api`

The API will be at **http://localhost:8000**


You can also open **http://localhost:8000/docs** to see the **Swagger** page and test the API from the browser.


To stop and remove the container:


 
`docker stop taka-api`

`docker rm taka-api`

**How to use the API**

The API has three **endpoints**:


**POST /predict** - send an image, get the detection result

**GET /** - basic info and class list

**GET /health** - check if the API is alive

Using curl:


 
`curl -X POST "http://localhost:8000/predict" -F "file=@test_images/test_100_taka.png"`

Using Postman:


Set method to POST and URL to **http://localhost:8000/predict**

Go to Body tab and select form-data

Add a key named "file", change its type from Text to File

Select an image file in the value

Click Send

Example response:


`{

    "count": 1,
    
    "image_size": {
    
        "width": 256,
        
        "height": 117
        
    },
    
    "detections": [
    
        {
        
            "class_id": 4,
            
            "class_name": "50_taka",
            
            "confidence": 0.9779,
            
            "bbox": {
            
                "x_min": 0.2,
                
                "y_min": 0.59,
                
                "x_max": 255.85,
                
                "y_max": 117.0
                
            }
            
        }
        
    ],
    
    "filename": "test_50_taka.png"
    
}`

**Error handling**

The API returns proper status codes for bad input:


**200** means **success**

**415** means **not an image**

**422** means **no files attached**

**503** means **the model file is missing**


**Requirements**

The Python packages are in requirements.txt: **ultralytics**, **fastapi**, **uvicorn[standard]**, **python-multipart**, **pillow**. **PyTorch** (CPU version) is installed inside the **Dockerfile**.
