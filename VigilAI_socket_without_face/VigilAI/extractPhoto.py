import os
video_name = "D:/VigilAI (2)/VigilAI/healthcare/healthcare/Eyebase/NA_2023_11_04+14_44_25_421505.webm"
video_folder = video_name.split("/")[-1]
video_folder = video_folder.split(".")[0]
file_paths = []
for filename in os.listdir(f"face_dataset/{video_folder}"):
    file_path = os.path.join(f"{os.getcwd()}/face_dataset/{video_folder}", filename)
    file_paths.append(file_path)
print(file_paths)