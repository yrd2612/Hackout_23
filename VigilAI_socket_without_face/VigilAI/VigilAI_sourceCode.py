#!/usr/bin/env python
# coding: utf-8

# In[73]:


import numpy as np
import tensorflow as tf
import os
import cv2
import random
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from collections import deque
import datetime
import json
import csv
import shutil
from urllib.request import urlopen
import time
import asyncio
import websockets
import base64
import os
import json
import csv



# In[74]:


global face_file_name
image_height, image_width = 64, 64
max_images_per_class = 8000
 
dataset_directory = "dataset"

classes_list = ["Fighting","Shooting","RoadAccidents","Robbery","Abuse","Arrest","Arson","Assault","Burglary","Explosion","Normal"]

 
model_output_size = len(classes_list)
print(model_output_size)


# In[75]:


model = tf.keras.models.load_model("./model_8000.h5", compile=False)


# In[76]:


indian_states = {
    "Andhra Pradesh": "AP",
    "Arunachal Pradesh": "AR",
    "Assam": "AS",
    "Bihar": "BR",
    "Chhattisgarh": "CG",
    "Goa": "GA",
    "Gujarat": "GJ",
    "Haryana": "HR",
    "Himachal Pradesh": "HP",
    "Jharkhand": "JH",
    "Karnataka": "KA",
    "Kerala": "KL",
    "Madhya Pradesh": "MP",
    "Maharashtra": "MH",
    "Manipur": "MN",
    "Meghalaya": "ML",
    "Mizoram": "MZ",
    "Nagaland": "NL",
    "Odisha": "OD",
    "Punjab": "PB",
    "Rajasthan": "RJ",
    "Sikkim": "SK",
    "Tamil Nadu": "TN",
    "Telangana": "TS",
    "Tripura": "TR",
    "Uttar Pradesh": "UP",
    "Uttarakhand": "UK",
    "West Bengal": "WB",
    "Na": "NA"
}

# # Example: Accessing the short form of a state
# print(indian_states["Tamil Nadu"])  # Output: "TN"


# In[77]:


def id(directory,state_name):
    id_count = len(os.listdir(directory))
    input_string = f"{state_name}"
    capitalized_string = input_string.title()
    print(capitalized_string)
    # print(capitalized_string)  # Output: "Hello World"

    id_number = indian_states[f"{capitalized_string}"]
    final_id_number = (f"{id_number}{id_count}")
    
    return final_id_number


# In[78]:


def upscale(image):
	sr = cv2.dnn_superres.DnnSuperResImpl_create()
	scale = 2
	modelPath = r"EDSR_x4.pb"
	method = "EDSR"
	sr.readModel(modelPath)
	sr.setModel(method.lower(), scale) 

	# Upscale the input image.
	result = sr.upsample(image) 

	return result


# In[79]:


# def extract_faces_from_stream(frame):
#     try:
#         print("calling extract faces 1")
#         model_path = "weights.caffemodel"  # Replace with the path to your model
#         proto_path = os.path.join(model_path, "deploy.prototxt.txt")
#         weight_path = os.path.join(model_path, "res10_300x300_ssd_iter_140000.caffemodel")
#         net = cv2.dnn.readNetFromCaffe("deploy_prototxt.txt","weights.caffemodel")

#         # Convert the image to a numpy array
#         # image = cv2.imdecode(np.frombuffer(image_stream.read(), np.uint8), cv2.IMREAD_COLOR)
#         # image = image_stream

#         # Get the dimensions of the image
#         (h, w) = frame.shape[:2]

#         # Preprocess the image for face detection
#         blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104, 177, 123))

#         # Set the input to the pre-trained model and perform face detection
#         net.setInput(blob)
#         detections = net.forward()

#         faces = []
#         for i in range(0, detections.shape[2]):
#             confidence = detections[0, 0, i, 2]

#             # Filter out weak detections
#             if confidence > 0.2:
#                 # Get the coordinates of the bounding box
#                 box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#                 (startX, startY, endX, endY) = box.astype("int")

#                 # Crop the face from the image
#                 face = frame[startY:endY, startX:endX]
#                 face = upscale(face)
#                 faces.append(face)
#         # print("len of faces at extract 1 is")
#             # cv2.imwrite(f"{i}_up.jpg",face)
#         print("len of faces at extract 1 is",len(faces))
#         len_faces = len(faces)
#     except:
#         pass
#         len_faces = 0
#         print("running except")

#     return len_faces


# In[80]:


import cv2
import os
import numpy as np 


# In[81]:

def extract_name_from_path(file_path):
    # Get the base filename from the file path
    base_filename = os.path.basename(file_path)
    
    # Remove the file extension from the base filename
    name_without_extension = os.path.splitext(base_filename)[0]
    
    return name_without_extension

def extract_faces_from_stream(frame):
    # try:
    print("calling extract faces 1")
    model_path = "weights.caffemodel"  # Replace with the path to your model
    proto_path = os.path.join(model_path, "deploy.prototxt.txt")
    weight_path = os.path.join(model_path, "res10_300x300_ssd_iter_140000.caffemodel")
    net = cv2.dnn.readNetFromCaffe("deploy_prototxt.txt","weights.caffemodel")

    # Convert the image to a numpy array
    # image = cv2.imdecode(np.frombuffer(image_stream.read(), np.uint8), cv2.IMREAD_COLOR)
    # image = image_stream

    # Get the dimensions of the image
    (h, w) = frame.shape[:2]

    # Preprocess the image for face detection
    blob = cv2.dnn.blobFromImage(frame, 1.0, (h,w), (104, 177, 123))

    # Set the input to the pre-trained model and perform face detection
    net.setInput(blob)
    detections = net.forward()

    faces = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > 0.2:
            # Get the coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Crop the face from the image
            face = frame[startY:endY, startX:endX]
            # face = upscale(face)
            faces.append(face)
    # print("len of faces at extract 1 is")
        # cv2.imwrite(f"{i}_up.jpg",face)
    print("len of faces at extract 1 is",len(faces))
    len_faces = len(faces)
    # except:
        
    #     len_faces = 0
    #     print("running except")

    return len_faces


# In[82]:


# frame = cv2.imread(r"C:/Users/YASH/Downloads/istockphoto-1480574526-170667a.jpg")
# extract_faces_from_stream(frame)


# In[83]:


# def extract_faces_from_stream2(frame,filename):
#     print("hello")
#     try:
#         photo_loc = []
#         frame = cv2.resize(frame,(240,320))
#         model_path = "weights.caffemodel"  # Replace with the path to your model
#         proto_path = os.path.join(model_path, "deploy.prototxt.txt")
#         weight_path = os.path.join(model_path, "res10_300x300_ssd_iter_140000.caffemodel")
#         net = cv2.dnn.readNetFromCaffe("deploy_prototxt.txt","weights.caffemodel")

#         # Convert the image to a numpy array
#         # image = cv2.imdecode(np.frombuffer(image_stream.read(), np.uint8), cv2.IMREAD_COLOR)
#         # image = image_stream

#         # Get the dimensions of the image
#         print(frame.shape)
#         (h, w) = frame.shape[:2]

#         # Preprocess the image for face detection
#         blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104, 177, 123))

#         # Set the input to the pre-trained model and perform face detection
#         net.setInput(blob)
#         detections = net.forward()

#         faces = []
#         for i in range(0, detections.shape[2]):
#             confidence = detections[0, 0, i, 2]

#             # Filter out weak detections
#             if confidence > 0.1:
#                 # Get the coordinates of the bounding box
#                 box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#                 (startX, startY, endX, endY) = box.astype("int")

#                 # Crop the face from the image
#                 face = frame[startY-10:endY+3, startX-5:endX+3]
#                 # face = upscale(face)
#                 faces.append(face)
                
#                 save_file_name = "./face_dataset/"+ str(filename) + str(i) + ".jpg"
#                 # save_file = f"C:/Users/rudra/Downloads/VigilAI (2)/VigilAI{save_file_name[1::]}"
#                 photo_loc.append(save_file_name)
#                 cv2.imwrite(save_file_name,face)
#         # photo_loc = []
#         print("length of faces is ",len(faces))
#         len_faces = len(faces)
#         if len(faces) == 0:
#             photo_loc = ['./0_up.jpg']
#     except:
#         len_faces = 0
#         photo_loc = ['./0_up.jpg']

#     return len_faces,photo_loc


# In[84]:


def extract_faces_from_stream2(frameMax,filename,max_index):
    print("hello")
    # try:
    photo_loc = []
    frame = cv2.resize(frameMax,(240,320))
    model_path = "weights.caffemodel"  # Replace with the path to your model
    proto_path = os.path.join(model_path, "deploy.prototxt.txt")
    weight_path = os.path.join(model_path, "res10_300x300_ssd_iter_140000.caffemodel")
    net = cv2.dnn.readNetFromCaffe("deploy_prototxt.txt","weights.caffemodel")

    # Convert the image to a numpy array
    # image = cv2.imdecode(np.frombuffer(image_stream.read(), np.uint8), cv2.IMREAD_COLOR)
    # image = image_stream

    # Get the dimensions of the image
    print(frame.shape)
    (h, w) = frame.shape[:2]

    # Preprocess the image for face detection
    blob = cv2.dnn.blobFromImage(frame, 1.0, (h,w), (104, 177, 123))

    # Set the input to the pre-trained model and perform face detection
    net.setInput(blob)
    detections = net.forward()

    faces = []
    video_folder = filename.split(".")[0]
    curr_dir =os.getcwd()
    video_folder_path = os.path.join(curr_dir,"face_dataset",video_folder)
    os.makedirs(video_folder_path,exist_ok=True)
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > 0.1:
            # Get the coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Crop the face from the image
            face = frame[startY-10:endY+3, startX-5:endX+3]
            # face = upscale(face)
            faces.append(face)
            
            save_file_name = "./face_dataset/"+ str(filename) + str(i) + ".jpg"
            # save_file = f"C:/Users/rudra/Downloads/VigilAI (2)/VigilAI{save_file_name[1::]}"
            photo_loc.append(save_file_name)
            # cv2.imwrite(save_file_name,face)
            # curr_dir = os.getcwd()
            # filename_folder = os.makedirs(filename.split(".")[0],exist_ok=True)
            # filename_folder_joined = os.path.join(curr_dir,"face_dataset",filename_folder,filename)
            # video_folder = filename.split(".")[0]
            # video_folder_path = os.path.join(curr_dir,"face_dataset",video_folder)
            # os.makedirs(video_folder_path,exist_ok=True)


            max_index = max_index/15
            if face is not None and not face.size == 0:
                save_file_name_ = os.path.join(video_folder_path,f"{max_index}.jpg")
            # save_file_name = os.path.join("face_dataset", f"{filename}{i}.jpg")
            # photo_loc.append(save_file_name)
            # cv2.imwrite(save_file_name, face)
                cv2.imwrite(save_file_name_,face)
            else:
                print("empty")
    # photo_loc = []
    print("length of faces is ",len(faces))
    len_faces = len(faces)
        
    # except:
    #     print("except for extract 2")
    #     len_faces = 0
    #     photo_loc = ['./0_up.jpg']

    return len_faces,photo_loc


# In[85]:


def save_clip(frames):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc('V','P','8','0')  
    frame_rate=15
    output_video = cv2.VideoWriter("./output_clip.webm", fourcc, frame_rate, (width, height))

    for frame in frames:
        output_video.write(frame)
    output_video.release()


# In[86]:


def get_location():
    try:
        # Create a geocoder instance
        url='http://ipinfo.io/json'
        response=urlopen(url)
        # Get the current GPS location
        data=json.load(response)
        coordinates=data['loc']
        city=data['city']
        state = data['region']
        city = 'Prayagraj'
        coordinates = '25.432227, 81.770702'
        state = 'Uttar Pradesh'
        return city,coordinates,state
    except:
        city,coordinates,state='NA','NA','NA'
        return city,coordinates,state


# In[87]:


def get_date_and_time():
    data = datetime.datetime.now()
    data=str(data)
    date,time=data.split(' ')
    date=date.replace('-','_')
    time=time.replace(':','_')
    time=time.replace('.','_')
    return str(date),str(time)


# In[88]:


# def create_database(crime_type,frames):
#     global face_file_name
#     # Specify the current file path and the new file name
#     current_file_path = "./healthcare/healthcare/Eyebase/output_clip.webm"
#     city,coordinates,state=get_location()

#     date,time=get_date_and_time()

#     new_file_name =str(city)+'_'+date+'+'+time+'.webm'
#     face_file_name = new_file_name
#     file_path='Eyebase/'+new_file_name

#     frames_in_videos = len(frames)
#     counter = 0
#     frames_in_videos = frames_in_videos//2
#     frame_list = []
#     frames_list = []
#     print(frames_in_videos)

#     for i,frame in enumerate(frames):
#         if i > frames_in_videos - 2 and  i < frames_in_videos + 2:
#                 number_of_faces = (extract_faces_from_stream(frame))
#                 # number_of_faces2 = len(number_of_faces)
#                 frame_list.append(number_of_faces)
#                 frames_list.append(frame)
#                 print(number_of_faces)
    
#     frame_list = np.array(frame_list)
#     max_index = np.argmax(frame_list)
#     print("max index is",max_index)
#     number_of_faces,photo_loc = (extract_faces_from_stream2(frames_list[max_index],new_file_name))

#     count_dir= "./healthcare/healthcare/Eyebase"
#     Fir_ID = id(count_dir,state)
#     new_data = [file_path,crime_type,city,coordinates,date,time,photo_loc,Fir_ID]

#     csv_file = "./healthcare/healthcare/AIbase.csv"

#     with open(csv_file, 'a', newline='',encoding='utf-8') as file:
#         writer = csv.writer(file)
#         # Append the new data to the CSV file
#         writer.writerow(new_data)

#     # Extract the directory and the extension from the current file path
#     directory = os.path.dirname(current_file_path)
#     extension = os.path.splitext(current_file_path)[1]

#     # Create the new file path by combining the directory, new file name, and extension
#     new_file_path = os.path.join(directory, new_file_name)

#     # Rename the file
#     os.rename(current_file_path, new_file_path)
#     print("done")


# In[89]:


# Function to read CSV data from a file
def read_csv_data(csv_file_path):
    with open(csv_file_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        csv_data = [row for row in csv_reader]
    return csv_data

# Function to read photo filenames from a folder
def get_photo_filenames(photos_folder):
    return [f"{photos_folder}/{filename}" for filename in os.listdir(photos_folder) if filename.endswith('.jpg') or filename.endswith('.png')]

# Function to encode video data as base64 string
def video_to_base64(video_file_path):
    with open(video_file_path, 'rb') as video_file:
        video_data = video_file.read()
        video_base64 = base64.b64encode(video_data).decode('utf-8')
    return video_base64


def encode_images_to_base64(photo_paths):
    encoded_images = []
    for photo_path in photo_paths:
        print(photo_path)
        with open(photo_path, 'rb') as image_file:
            binary_image_data = image_file.read()
            # Encode binary data as base64
            base64_image_data = base64.b64encode(binary_image_data).decode('utf-8')
            encoded_images.append(base64_image_data)
    return encoded_images



import asyncio
import websockets

async def send_video(path):
    video_filename = path

    # photos_filename = video_filename.split("/")[-1]
    photos_filename = extract_name_from_path(video_filename)
    photos_filename = f"face_dataset/{photos_filename}"
    chunk_size = 1024*500

    csv_data = read_csv_data("healthcare/healthcare/AIbase.csv")  # Replace with the path to your CSV file
    photo_filenames = get_photo_filenames(photos_filename)  # Replace with the path to your photos folder
    video_base64 = video_to_base64(video_filename)  # Replace with the path to your video file
    photos_encoded = encode_images_to_base64(photo_filenames)
    photo_filenames= [f'{photos_filename}']
    print("len of photo encoding is",len(photos_encoded))
    data = {
        "csv": csv_data,
        "photos": photo_filenames,
        "photo_base64":photos_encoded,
        "video": video_base64
    }


    
    print("srartingg")
    # async with websockets.connect('ws://192.168.25.171:8765') as websocket: 
    async with websockets.connect('ws://192.168.165.57:8765') as websocket:
        print("here") 
        json_data = json.dumps(data)
        await websocket.send(json_data)
        print("sent")



# In[90]:

def create_csv_file(file_path, data):
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)


def create_database(crime_type,frames):
    global face_file_name
    # Specify the current file path and the new file name
    current_file_path = "./healthcare/healthcare/Eyebase/output_clip.webm"
    city,coordinates,state=get_location()

    date,time=get_date_and_time()

    new_file_name =str(city)+'_'+date+'+'+time+'.webm'
    face_file_name = new_file_name
    file_path='Eyebase/'+new_file_name

    frames_in_videos = len(frames)
    counter = 0
    frames_in_videos = frames_in_videos//2
    frame_list = []
    frames_list = []
    print(frames_in_videos)

    for i,frame in enumerate(frames):
        if i > frames_in_videos - 5 and  i < frames_in_videos + 5:
                number_of_faces = (extract_faces_from_stream(frame))
                # number_of_faces2 = len(number_of_faces)
                frame_list.append(number_of_faces)
                frames_list.append(frame)
                print(number_of_faces)
    # for frame in frames:
    #     number_of_faces = extract_faces_from_stream(frame)
    #     frame_list.append(number_of_faces)

        
    
    frame_list = np.array(frame_list)
    print("frame list is",frame_list)
    max_index = np.argmax(frame_list)
    print("max index is",max_index)
    number_of_faces,photo_loc = extract_faces_from_stream2(frames[max_index],new_file_name,max_index)

    count_dir= "./healthcare/healthcare/Eyebase"
    Fir_ID = id(count_dir,state)
    new_data = [file_path,crime_type,city,coordinates,date,time,photo_loc,Fir_ID]

    csv_file = "./healthcare/healthcare/AIbase.csv"

    with open(csv_file, 'a', newline='',encoding='utf-8') as file:
        writer = csv.writer(file)
        # Append the new data to the CSV file
        writer.writerow(new_data)

    # Extract the directory and the extension from the current file path
    directory = os.path.dirname(current_file_path)
    extension = os.path.splitext(current_file_path)[1]

    # Create the new file path by combining the directory, new file name, and extension
    new_file_path = os.path.join(directory, new_file_name)

    # Rename the file
    os.rename(current_file_path, new_file_path)
    path = new_file_path
    asyncio.run(send_video(path))
    data=[
        ['Name','Crime_Type','Location','Co-ordinates','Date','Time',"photo_loc","fir"]
    ]

    create_csv_file("healthcare/healthcare/AIbase.csv",data)
    print("done")


# In[ ]:





# In[91]:


def add_to_database(crime_type,frames):
    # Specify the source file path and the destination folder path
    source_file = "./output_clip.webm"
    destination_folder = "./healthcare/healthcare/Eyebase"

    # Copy the file to the destination folder
    shutil.copy(source_file, destination_folder)
    create_database(crime_type,frames)


# In[92]:


# def predict_on_live_video(video_file_path, output_file_path, window_size):
#     frame_count=0
#     frames=[]
#     crime_type='NA'
#     counter=0
#     # Initialize a Deque Object with a fixed size which will be used to implement moving/rolling average functionality.
#     predicted_labels_probabilities_deque = deque(maxlen = window_size)
 
#     # Reading the Video File using the VideoCapture Object
# #     video_reader = cv2.VideoCapture(video_file_path)
#     video_reader = cv2.VideoCapture(video_file_path)
 
#     # Getting the width and height of the video 
#     original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
#     original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
 
#     # Writing the Overlayed Video Files Using the VideoWriter Object
#     # video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc(*'VP80'), 24, (original_video_width, original_video_height))
 
#     while True: 
 
#         # Reading The Frame
#         status, frame = video_reader.read() 
 
#         if not status:
#             break
 
#         # Resize the Frame to fixed Dimensions
#         resized_frame = cv2.resize(frame, (image_height, image_width))
         
#         # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
#         normalized_frame = resized_frame / 255
 
#         # Passing the Image Normalized Frame to the model and receiving Predicted Probabilities.
#         predicted_labels_probabilities = model.predict(np.expand_dims(normalized_frame, axis = 0))[0]
 
#         # Appending predicted label probabilities to the deque object
#         predicted_labels_probabilities_deque.append(predicted_labels_probabilities)
 
#         # Assuring that the Deque is completely filled before starting the averaging process
#         if len(predicted_labels_probabilities_deque) == window_size:
 
#             # Converting Predicted Labels Probabilities Deque into Numpy array
#             predicted_labels_probabilities_np = np.array(predicted_labels_probabilities_deque)
 
#             # Calculating Average of Predicted Labels Probabilities Column Wise 
#             predicted_labels_probabilities_averaged = predicted_labels_probabilities_np.mean(axis = 0)
 
#             # Converting the predicted probabilities into labels by returning the index of the maximum value.
#             predicted_max=np.max(predicted_labels_probabilities_averaged)

#             predicted_label = np.argmax(predicted_labels_probabilities_averaged)

#             if predicted_max>0.6 and predicted_label<10:
#                 predicted_label = np.argmax(predicted_labels_probabilities_averaged)
#                 frame_count=frame_count+1
#                 if frame_count<450:
#                     if counter==0:
#                         crime_type=classes_list[predicted_label]
#                         counter=1
#                     frames.append(frame)
#                     save_clip(frames)
#                 else:
#                     if len(frames)>100:    
#                         frame_count=0
#                         frames=[]
#             else:
#                 predicted_label=10 

                
#             # Accessing The Class Name using predicted label.
#             predicted_class_name = classes_list[predicted_label]

           
#         # Writing The Frame
#         # video_writer.write(frame)
 
 
#         cv2.imshow('CCTV FOOTAGE', frame)
 
#         key_pressed = cv2.waitKey(1)
 
#         if key_pressed == ord('q'):
#             break
            
#     add_to_database(crime_type)
#     cv2.destroyAllWindows()
     
#     # Closing the VideoCapture and VideoWriter objects and releasing all resources held by them. 
#     video_reader.release()
#     # video_writer.release()


# In[93]:


# count=0
import threading
# frame_save = []

class test:
    def __init__ (self,video_file_path):
        self.frame_count = 0 
        self.frames = []
        self.status=0
        self.crime_type = "NA"
        self.counter = 0
        self.video_file_path=video_file_path
        self.predicted_labels_probabilities_deque = deque(maxlen = 30)
        self.video_reader = cv2.VideoCapture(self.video_file_path)
        self.processing_thread = None
        self.is_processing = False
        self.window_size=30
        self.ti=None
        self.count=0
        self.lock=threading.Lock()
        self.t3=None
        self.start_time = time.time()
        self.end_time = 0


    def start_processing(self):
        print(self.video_file_path)
        self.is_processing = True
        print("gbg")
        # video_reader = cv2.VideoCapture(self.video_file_path)
        self.processing_thread = threading.Thread(target=self.process_video)
        self.processing_thread.start()
        print(self.video_file_path)
        # print(video_reader)
        # self.status,self.frames = video_reader.read()
        print(self.status)
        # return frames

    def stop_processing(self):
        self.is_processing = False
        if self.processing_thread is not None:
            # self.processing_thread.join()
            pass
    
    def process_video(self):
        
        # original_video_width = int(self.frames(cv2.CAP_PROP_FRAME_WIDTH))
        # original_video_height = int(self.frames(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print("kkk")
        counter=0
        while self.is_processing:
            self.lock.acquire()
            print("lll")
            frames=self.frames
            status, frame = self.video_reader.read()
            if not status:
                print(self.status)
                break
            print("mmm")
            resized_frame = cv2.resize(frame, (image_height, image_width))
            normalized_frame = resized_frame / 255
            # self.processing_thread.join()
            self.lock.release()
            self.ti=threading.Thread(target=test.predict_on_live_video,args= [self,normalized_frame,frame,counter])
            self.ti.start()

            # ti.join()
            # predicted_labels_probabilities = model.predict(np.expand_dims(normalized_frame, axis = 0))[0]
            # self.predicted_labels_probabilities_deque.append(predicted_labels_probabilities)
            # if len(self.predicted_labels_probabilities_deque) == self.window_size:
            #     predicted_labels_probabilities_np = np.array(self.predicted_labels_probabilities_deque)
            #     predicted_labels_probabilities_averaged = predicted_labels_probabilities_np.mean(axis = 0)
            #     predicted_max=np.max(predicted_labels_probabilities_averaged)
            #     predicted_label = np.argmax(predicted_labels_probabilities_averaged)
            #     if predicted_max>0.6 and predicted_label<10:
            #         predicted_label = np.argmax(predicted_labels_probabilities_averaged)
            #         self.frame_count = self.frame_count + 1
            #         if frame_count<450:
            #             if counter == 0:
            #                 crime_type = classes_list[predicted_label]
            #                 counter = 1
            #             frames.append(frame)
            #             save_clip(frames)
            #         else:
            #             if len(frames)>100:
            #                 frame_count = 0
            #                 frames = []
            #     else:
            #         predicted_label= 10
            #     predicted_class_name = classes_list[predicted_label]
            cv2.imshow('CCTV FOOTAGE',frame)
            key_pressed = cv2.waitKey(1)
            if key_pressed == ord('q'):
                break
        cv2.destroyAllWindows()
        # self.t3 = threading.Thread(target=save_clip,args=[self.frames])
        # self.t3.start()
        save_clip(self.frames)
        self.end_time = time.time()
        print("time==",self.end_time-self.start_time)
        time.sleep(3)
        add_to_database(self.crime_type,self.frames)
        # self.t3.join()
        print(threading.active_count())
            # video_reader.release()
        # t1 = threading.Thread(target=read_video,args=[])
        # t2 = threading.Thread(target=process_video,args=[])
        # t1.start()
        # t2.start()
        # t1.join()
        # t2.join()
    def predict_on_live_video(self,normalized_frame,frame,counter,window_size=30):
        # t3 = threading.Thread(target=save_clip,args=[self.frames])
        self.lock.acquire()
        # predicted_labels_probabilities_deque = deque(maxlen = window_size
        # print("predict",threading.active_count())
        predicted_labels_probabilities = model.predict(np.expand_dims(normalized_frame, axis = 0))[0]
        self.predicted_labels_probabilities_deque.append(predicted_labels_probabilities)
        if len(self.predicted_labels_probabilities_deque) == self.window_size:
                predicted_labels_probabilities_np = np.array(self.predicted_labels_probabilities_deque)
                predicted_labels_probabilities_averaged = predicted_labels_probabilities_np.mean(axis = 0)
                predicted_max=np.max(predicted_labels_probabilities_averaged)
                predicted_label = np.argmax(predicted_labels_probabilities_averaged)
                if predicted_max>0.6 and predicted_label<10:
                    predicted_label = np.argmax(predicted_labels_probabilities_averaged)
                    self.frame_count = self.frame_count + 1
                    print("counter",self.counter)
                    if self.frame_count<450:
                        print("frame_count",self.frame_count)
                        if self.counter == 0:
                            print("ayaya?")
                            self.crime_type = classes_list[predicted_label]
                            self.counter = 1
                        self.frames.append(frame)
                        # trying to save clip in a new thread
                        # save_clip(self.frames)
                        frame_save = self.frames
                        # t3.start()
                    else:
                        if len(self.frames)>100:
                            self.frame_count = 0
                            # self.frames = []
                else:
                    self.frame_count = 0
                    predicted_label= 10
                predicted_class_name = classes_list[predicted_label]
        else:
            self.frames=[]
        self.lock.release() 
        # self.ti.join()   
        # self.count=1
        # t2 = threading.Thread(target=test.process_video,args=[self])
        # t1 = threading.Thread(target=test.read_video,args=[self])
        # print(self.count)
        # t2.start()
        # t1.start()
        # # t1.join()
        # t2.join()
        




# In[94]:


# import tkinter as tk
# from tkinter.filedialog import askopenfilename
# root = tk.Tk()
# output_file_path = "C:/Users/rudra/OneDrive/Desktop"
# window_size = 30
# video_file_path = 0
# def start():
#     file_path = askopenfilename()
#     output_file_path = "C:/Users/rudra/OneDrive/Desktop"
#     window_size = 30
#     print(file_path)
#     video_file_path = file_path
#     predict_on_live_video(video_file_path, output_file_path, window_size)
# predict_on_live_video(video_file_path, output_file_path, window_size)
# # start()


# In[95]:


# import tkinter as tk
# from tkinter.filedialog import askopenfilename
# def button_click():
#     file_path = askopenfilename()

#     # Display the selected file path
#     # print("Selected file path:", file_path)
#     start(file_path)

# # Create the Tkinter root window
# button = tk.Button(root, text="Click Me!",command = start)
# button.pack()
# root.mainloop()
import tkinter as tk
from tkinter.filedialog import askopenfilename
root = tk.Tk()
root.geometry("600x600")
# output_file_path = "C:/Users/rudra/OneDrive/Desktop"
# window_size = 30
# video_file_path = 0
def start():
    # print("tkinter",threading.active_count())
    file_path = askopenfilename()
    output_file_path = "C:/Users/dwive/Desktop"
    window_size = 30
    print(file_path)
    video_file_path = file_path
    my_object = test(video_file_path)
    # print("start")
    my_object.start_processing()
def start_camera():
    file_path = 0
    output_file_path = "C:/Users/dwive/Desktop"
    window_size = 30
    print(file_path)
    video_file_path = file_path
    my_object = test(video_file_path)
    # print("start")
    my_object.start_processing()

# predict_on_live_video(video_file_path, output_file_path, window_size)
# start()


# In[96]:


# import tkinter as tk
# from tkinter.filedialog import askopenfilename
def button_click():
    file_path = askopenfilename()

    # Display the selected file path
    # print("Selected file path:", file_path)
    start(file_path)

# def cam():
#     cam_start(0)

# Create the Tkinter root window
button = tk.Button(root, text="Click Me!",command = start)
button1 = tk.Button(root, text="Camera!",command = start_camera)
button.pack()
button1.pack()
root.mainloop()

# # tkinter good button
# button = tk.Button(root,font="comicsans 35 bold",bg="yellow",padx=30, text="Click Me!",command = start)
# button.place(x= 100,y=300)
# button.pack(pady= 120)

# button2 = tk.Button(root,font="comicsans 35 bold",bg="yellow",padx=30, text="Live Cam!",command = cam_start)
# button2.place(x=100,y=400)
# button.pack(pady= 120)
# root.mainloop()


# In[ ]:





# In[ ]:




