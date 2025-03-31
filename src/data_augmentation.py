import os
import cv2
import numpy as np
import imgaug.augmenters as iaa
from tqdm import tqdm

def augment_images(input_dir, output_dir, augmentation_factor=3):
    """
    Tăng cường dữ liệu bằng cách tạo ra các biến thể của ảnh
    """
    # Tạo thư mục đầu ra
    os.makedirs(output_dir, exist_ok=True)
    
    # Định nghĩa các phép biến đổi
    augmenter = iaa.Sequential([
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
        iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(scale=(0, 0.05*255))),
        iaa.Sometimes(0.5, iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)
        )),
        iaa.Sometimes(0.5, iaa.PerspectiveTransform(scale=(0.01, 0.1))),
        iaa.Sometimes(0.5, iaa.Multiply((0.8, 1.2))),
        iaa.Sometimes(0.5, iaa.LinearContrast((0.75, 1.5)))
    ])
    
    # Chỉ tăng cường dữ liệu cho lớp 40km vì số lượng ít hơn
    input_dir_40km = os.path.join(input_dir, "train", "40km")
    output_dir_40km = os.path.join(output_dir, "40km")
    os.makedirs(output_dir_40km, exist_ok=True)
    
    # Copy ảnh gốc
    image_files = os.listdir(input_dir_40km)
    for file in image_files:
        img_path = os.path.join(input_dir_40km, file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Lưu ảnh gốc
        cv2.imwrite(os.path.join(output_dir_40km, file), img)
        
        # Tạo các biến thể
        for i in range(augmentation_factor):
            img_aug = augmenter(image=img)
            filename, ext = os.path.splitext(file)
            new_filename = f"{filename}_aug_{i}{ext}"
            cv2.imwrite(os.path.join(output_dir_40km, new_filename), img_aug)
    
    print(f"Đã tăng cường dữ liệu: {len(image_files)} ảnh gốc -> {len(image_files) * (augmentation_factor + 1)} ảnh")

if __name__ == "__main__":
    processed_data_dir = "../data/processed"
    augmented_data_dir = "../data/augmented"
    augment_images(processed_data_dir, augmented_data_dir)
