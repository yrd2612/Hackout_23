import numpy as np
import tensorflow as tf
import os
import cv2
import random
import pandas as pd
from collections import deque
import asyncio
import websockets
seed_constant = 15
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)
image_height, image_width = 64, 64
max_images_per_class = 8000
 
dataset_directory = "dataset"

classes_list = ["Fighting","Shooting","RoadAccidents","Robbery","Abuse","Arrest","Arson","Assault","Burglary","Explosion","Normal"]

 
model_output_size = len(classes_list)
model = tf.keras.models.load_model('model_8000.h5',compile=False)
def save_clip(frames):
    # print(frames)
    height, width, _ = frames[0].shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  
    frame_rate=30
    output_video = cv2.VideoWriter("output_clip.mp4", fourcc, frame_rate, (width, height))

    for frame in frames:
        output_video.write(frame)
    output_video.release()
 
def predict_on_live_video(video_file_path, output_file_path, window_size):
    frame_count=0
    frames=[]
    # Initialize a Deque Object with a fixed size which will be used to implement moving/rolling average functionality.
    predicted_labels_probabilities_deque = deque(maxlen = window_size)
 
    # Reading the Video File using the VideoCapture Object
#     video_reader = cv2.VideoCapture(video_file_path)
    video_reader = cv2.VideoCapture(video_file_path)
 
    # Getting the width and height of the video 
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
 
    # Writing the Overlayed Video Files Using the VideoWriter Object
    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 24, (original_video_width, original_video_height))
 
    while True: 
 
        # Reading The Frame
        status, frame = video_reader.read() 
 
        if not status:
            break
 
        # Resize the Frame to fixed Dimensions
        resized_frame = cv2.resize(frame, (image_height, image_width))
         
        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
        normalized_frame = resized_frame / 255
 
        # Passing the Image Normalized Frame to the model and receiving Predicted Probabilities.
        predicted_labels_probabilities = model.predict(np.expand_dims(normalized_frame, axis = 0))[0]
 
        # Appending predicted label probabilities to the deque object
        predicted_labels_probabilities_deque.append(predicted_labels_probabilities)
 
        # Assuring that the Deque is completely filled before starting the averaging process
        if len(predicted_labels_probabilities_deque) == window_size:
 
            # Converting Predicted Labels Probabilities Deque into Numpy array
            predicted_labels_probabilities_np = np.array(predicted_labels_probabilities_deque)
 
            # Calculating Average of Predicted Labels Probabilities Column Wise 
            predicted_labels_probabilities_averaged = predicted_labels_probabilities_np.mean(axis = 0)
 
            # Converting the predicted probabilities into labels by returning the index of the maximum value.
            predicted_max=np.max(predicted_labels_probabilities_averaged)
            predicted_label = np.argmax(predicted_labels_probabilities_averaged)
            print(predicted_max)
            if predicted_max>0.8 and predicted_label<10:
                predicted_label = np.argmax(predicted_labels_probabilities_averaged)
                frame_count=frame_count+1
                if frame_count<450:
                    frames.append(frame)
                else:
                    frame_count=0
                    frames=[]
            else:
                predicted_label=10 
            # Accessing The Class Name using predicted label.
            predicted_class_name = classes_list[predicted_label]

           
            # Overlaying Class Name Text Ontop of the Frame
            # cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            print(predicted_class_name)
 
        # Writing The Frame
        video_writer.write(frame)
 
 
        cv2.imshow('Predicted Frames', frame)
 
        key_pressed = cv2.waitKey(10)
 
        if key_pressed == ord('q'):
            break
 
    cv2.destroyAllWindows()
    save_clip(frames)
 
     
    # Closing the VideoCapture and VideoWriter objects and releasing all resources held by them. 
    video_reader.release()
    video_writer.release()

def start_process(path):
    output_file_path = "."
    print("here")
    # output_file_path = 0
    window_size = 30
    video_file_path = path
    predict_on_live_video(video_file_path, output_file_path, window_size)



async def receive_video(websocket, path):
    video_filename = 'received_video.mp4'
    chunk_size = 4096
    path = "D:/hackathon/headout/Hackout_23/received_video.mp4"
    with open(video_filename, 'wb') as video_file:
        while True:
            try:
                chunk = await websocket.recv()
            except:
                break
            if not chunk:
                break
            video_file.write(chunk)
    start_process(path)

start_server = websockets.serve(receive_video, '0.0.0.0', 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()