import os
import shutil

def prepare_augmented_dataset():
    # Tạo thư mục mới cho dữ liệu huấn luyện
    data_dir = "../data"
    processed_aug_dir = os.path.join(data_dir, "processed_with_augmentation")
    train_dir = os.path.join(processed_aug_dir, "train")
    
    # Tạo cấu trúc thư mục
    os.makedirs(processed_aug_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(os.path.join(train_dir, "40km"), exist_ok=True)
    os.makedirs(os.path.join(train_dir, "non40km"), exist_ok=True)

    # Sao chép dữ liệu not_40km từ tập huấn luyện gốc
    original_not_40km = os.path.join(data_dir, "processed/train/non40km")
    for file in os.listdir(original_not_40km):
        shutil.copy2(
            os.path.join(original_not_40km, file),
            os.path.join(train_dir, "non40km", file)
        )

    # Sao chép dữ liệu 40km gốc và tăng cường
    original_40km = os.path.join(data_dir, "processed/train/40km")
    augmented_40km = os.path.join(data_dir, "augmented/40km")

    # Sao chép dữ liệu 40km gốc
    for file in os.listdir(original_40km):
        shutil.copy2(
            os.path.join(original_40km, file),
            os.path.join(train_dir, "40km", file)
        )

    # Sao chép dữ liệu 40km tăng cường
    for file in os.listdir(augmented_40km):
        # Chỉ sao chép file tăng cường (có chứa "_aug_")
        if "_aug_" in file:
            shutil.copy2(
                os.path.join(augmented_40km, file),
                os.path.join(train_dir, "40km", file)
            )

    # Sao chép nguyên tập validation và test
    for split in ["val", "test"]:
        dest_dir = os.path.join(processed_aug_dir, split)
        os.makedirs(dest_dir, exist_ok=True)
        os.makedirs(os.path.join(dest_dir, "40km"), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, "non40km"), exist_ok=True)
        
        # Sao chép từ dữ liệu gốc
        for cls in ["40km", "non40km"]:
            src_dir = os.path.join(data_dir, f"processed/{split}/{cls}")
            for file in os.listdir(src_dir):
                shutil.copy2(
                    os.path.join(src_dir, file),
                    os.path.join(dest_dir, cls, file)
                )

    # Đếm số lượng ảnh
    train_40km_count = len(os.listdir(os.path.join(train_dir, "40km")))
    train_not_40km_count = len(os.listdir(os.path.join(train_dir, "non40km")))
    val_40km_count = len(os.listdir(os.path.join(processed_aug_dir, "val/40km")))
    val_not_40km_count = len(os.listdir(os.path.join(processed_aug_dir, "val/non40km")))
    test_40km_count = len(os.listdir(os.path.join(processed_aug_dir, "test/40km")))
    test_not_40km_count = len(os.listdir(os.path.join(processed_aug_dir, "test/non40km")))
    
    print("=== Thống kê dữ liệu ===")
    print(f"Train - 40km: {train_40km_count} ảnh")
    print(f"Train - not_40km: {train_not_40km_count} ảnh")
    print(f"Val - 40km: {val_40km_count} ảnh")
    print(f"Val - not_40km: {val_not_40km_count} ảnh")
    print(f"Test - 40km: {test_40km_count} ảnh")
    print(f"Test - not_40km: {test_not_40km_count} ảnh")
    print("\nĐã chuẩn bị xong dữ liệu cho huấn luyện!")

if __name__ == "__main__":
    # Tạo thư mục models nếu chưa có
    os.makedirs("../models", exist_ok=True)
    
    # Chuẩn bị dữ liệu
    prepare_augmented_dataset()
