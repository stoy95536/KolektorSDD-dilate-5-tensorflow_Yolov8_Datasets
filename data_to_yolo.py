import os
from glob import glob


# 設定資料夾
image_folder_path = "./yolo_dataset"
output_dataset_path = "datasets"  # YOLO 資料集輸出資料夾

# 建立 YOLO 資料夾結構
os.makedirs(f"{output_dataset_path}/images/train", exist_ok=True)
os.makedirs(f"{output_dataset_path}/images/val", exist_ok=True)
os.makedirs(f"{output_dataset_path}/labels/train", exist_ok=True)
os.makedirs(f"{output_dataset_path}/labels/val", exist_ok=True)


image_folders = [f'{image_folder_path}/{i}' for i in os.listdir(image_folder_path)]

image_number = 0

num_train = int(48*0.8)

for image_folder in image_folders:

    
    for image in os.listdir(f"{image_folder}/images/test"):
        dataset_type = "train" if image_number < num_train else "val"
        filename = image.replace(".jpg", "")
        
        os.rename(f"{image_folder}/images/test/{filename}.jpg", f"{output_dataset_path}/images/{dataset_type}/{image_number:03}.jpg")        
        os.rename(f"{image_folder}/labels/test/{filename}.txt", f"{output_dataset_path}/labels/{dataset_type}/{image_number:03}.txt")        
        image_number += 1
        
    for image in os.listdir(f"{image_folder}/images/train"):
        dataset_type = "train" if image_number < num_train else "val"
        filename = image.replace(".jpg", "")
        
        os.rename(f"{image_folder}/images/train/{filename}.jpg", f"{output_dataset_path}/images/{dataset_type}/{image_number:03}.jpg")        
        os.rename(f"{image_folder}/labels/train/{filename}.txt", f"{output_dataset_path}/labels/{dataset_type}/{image_number:03}.txt")         
        image_number += 1


