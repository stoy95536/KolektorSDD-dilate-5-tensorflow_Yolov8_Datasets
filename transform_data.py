import os
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from PIL import Image

def decode_image(image_bytes):
    """ 解析圖片格式 """
    image = tf.image.decode_image(image_bytes, channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)
    return image.numpy()

def decode_mask(mask_bytes):
    """ 解析 segmentation mask（BMP/PNG） """
    mask = tf.image.decode_image(mask_bytes, channels=1)
    mask = tf.image.convert_image_dtype(mask, tf.uint8)
    return mask.numpy()

def extract_polygons(mask, min_area=5):
    """ 從 mask 提取 segmentation polygon """
    if len(mask.shape) == 3:  # 如果有多個通道，轉為單通道
        mask = mask[:, :, 0]
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = mask.shape
    polygons = []

    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        polygon = contour.reshape(-1, 2) / [width, height]  # 歸一化
        polygons.append(polygon.flatten().tolist())

    return polygons

def save_yolo_format(label_path, polygons):
    """ 儲存 YOLO segmentation 格式標註 """
    with open(label_path, 'w') as f:
        for polygon in polygons:
            f.write("0 " + " ".join(map(str, polygon)) + "\n")

def process_tfrecord(tfrecord_path, output_img_dir, output_lbl_dir):
    """ 解析 TFRecord 並轉換為 YOLOv8 Segmentation 格式 """
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_lbl_dir, exist_ok=True)
    
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    
    for raw_record in tqdm(dataset, desc=f"Processing {tfrecord_path}"):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        
        # 解析圖片與標註
        img_bytes = example.features.feature['image/encoded'].bytes_list.value[0]
        mask_bytes = example.features.feature['image/class/encoded'].bytes_list.value[0]
        filename = example.features.feature['image/class/filename'].bytes_list.value[0].decode()
        
        # 解析圖片並轉換為 JPG
        img = decode_image(img_bytes)
        img_filename = filename.rsplit('.', 1)[0] + '.jpg'
        img_path = os.path.join(output_img_dir, img_filename)
        Image.fromarray(img).save(img_path, quality=95)
        
        # 解析 segmentation mask 並提取 polygon
        mask = decode_mask(mask_bytes)
        polygons = extract_polygons(mask)
        
        # 儲存 YOLO 格式標註
        label_filename = img_filename.rsplit('.', 1)[0] + '.txt'
        label_path = os.path.join(output_lbl_dir, label_filename)
        save_yolo_format(label_path, polygons)

def process_all_folds(dataset_dir, output_dir):
    """ 處理 datasets/fold_0, fold_1, fold_2 """
    for fold in ['fold_0', 'fold_1', 'fold_2']:
        fold_path = os.path.join(dataset_dir, fold)
        for split in ['train', 'test']:
            tfrecord_path = os.path.join(fold_path, f'{split}-00000-of-00001')
            output_img_dir = os.path.join(output_dir, fold, f'images/{split}')
            output_lbl_dir = os.path.join(output_dir, fold, f'labels/{split}')
            process_tfrecord(tfrecord_path, output_img_dir, output_lbl_dir)

# 設定資料夾
DATASET_DIR = "./KolektorSDD-dilate=5-tensorflow/KolektorSDD-dilate=5/"  # TFRecord 資料夾
OUTPUT_DIR = "yolo_dataset"  # 轉換後的 YOLO 格式資料夾

# 執行轉換
process_all_folds(DATASET_DIR, OUTPUT_DIR)