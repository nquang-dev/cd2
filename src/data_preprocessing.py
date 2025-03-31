import os
import shutil
import random
from sklearn.model_selection import train_test_split
import cv2
import numpy as np

def split_dataset(raw_data_dir, processed_data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Chia dataset thành tập train, validation và test và in ra thông tin chi tiết
    """
    # Tạo thư mục
    for split in ['train', 'val', 'test']:
        for label in ['40km', 'non40km']:
            os.makedirs(os.path.join(processed_data_dir, split, label), exist_ok=True)
    
    # Xử lý từng lớp
    for label in ['40km', 'non40km']:
        # Lấy danh sách tất cả ảnh
        image_files = os.listdir(os.path.join(raw_data_dir, label))
        total_images = len(image_files)

        # Làm tròn số lượng ảnh
        train_size = round(total_images * train_ratio)
        val_size = round(total_images * val_ratio)
        test_size = total_images - train_size - val_size  # Đảm bảo tổng đúng 100%

        # Chia tập dữ liệu
        train_files, temp_files = train_test_split(image_files, train_size=train_size, random_state=42)
        val_files, test_files = train_test_split(temp_files, test_size=test_size, random_state=42)

        # Di chuyển file vào thư mục tương ứng
        for file in train_files:
            shutil.copy(os.path.join(raw_data_dir, label, file), os.path.join(processed_data_dir, 'train', label, file))
        
        for file in val_files:
            shutil.copy(os.path.join(raw_data_dir, label, file), os.path.join(processed_data_dir, 'val', label, file))
        
        for file in test_files:
            shutil.copy(os.path.join(raw_data_dir, label, file), os.path.join(processed_data_dir, 'test', label, file))

        # In kết quả phân chia
        print(f"[{label}] Tổng số ảnh: {total_images}")
        print(f" - Train: {train_size} ảnh ({train_ratio*100:.0f}%)")
        print(f" - Validation: {val_size} ảnh ({val_ratio*100:.0f}%)")
        print(f" - Test: {test_size} ảnh ({test_ratio*100:.0f}%)")
        print("-" * 40)

    print("Phân chia dữ liệu hoàn tất!")



def preprocess_image(image_path, target_size=(224, 224)):
    """
    Tiền xử lý ảnh: đọc, resize và chuẩn hóa
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Chuyển từ BGR sang RGB
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Chuẩn hóa giá trị pixel về khoảng [0, 1]
    return img

if __name__ == "__main__":
    raw_data_dir = "../data/raw"
    processed_data_dir = "../data/processed"
    split_dataset(raw_data_dir, processed_data_dir)
