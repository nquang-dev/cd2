import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import os

def load_tflite_model(model_path):
    """
    Tải mô hình TFLite
    """
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_image(image, target_size=(224, 224)):
    """
    Tiền xử lý ảnh đầu vào
    """
    # Resize
    image_resized = cv2.resize(image, target_size)
    # Chuẩn hóa
    image_normalized = image_resized / 255.0
    # Chuyển sang định dạng batch
    image_batch = np.expand_dims(image_normalized, axis=0)
    return image_batch

def detect_traffic_sign(interpreter, image):
    """
    Phát hiện biển báo giao thông từ ảnh
    """
    # Lấy thông tin đầu vào/ra
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Tiền xử lý ảnh
    input_data = preprocess_image(image)
    
    # Đặt tensor đầu vào
    interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))
    
    # Chạy suy luận
    start_time = time.time()
    interpreter.invoke()
    inference_time = time.time() - start_time
    
    # Lấy kết quả
    output_data = interpreter.get_tensor(output_details[0]['index'])
    score = output_data[0][0]
    
    # Phân loại
    is_40km_sign = score > 0.5
    
    return is_40km_sign, score, inference_time

def run_detection_on_image(model_path, image_path):
    """
    Chạy phát hiện trên một ảnh
    """
    # Tải mô hình
    interpreter = load_tflite_model(model_path)
    
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        print(f"Không thể đọc ảnh: {image_path}")
        return
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Phát hiện
    is_40km_sign, confidence, inference_time = detect_traffic_sign(interpreter, image_rgb)
    
    # Hiển thị kết quả
    result_text = f"40km/h: {'YES' if is_40km_sign else 'NO'}"
    confidence_text = f"Confidence: {confidence:.2f}"
    time_text = f"Time: {inference_time*1000:.1f}ms"
    
    cv2.putText(image, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(image, confidence_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(image, time_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Hiển thị ảnh
    cv2.imshow("Traffic Sign Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_detection_on_directory(model_path, image_dir):
    """
    Chạy phát hiện trên tất cả ảnh trong một thư mục
    """
    # Tải mô hình
    interpreter = load_tflite_model(model_path)
    
    # Lấy danh sách file ảnh
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"Không tìm thấy file ảnh nào trong thư mục {image_dir}")
        return
    
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        
        # Đọc ảnh
        image = cv2.imread(image_path)
        if image is None:
            print(f"Không thể đọc ảnh: {image_path}")
            continue
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Phát hiện
        is_40km_sign, confidence, inference_time = detect_traffic_sign(interpreter, image_rgb)
        
        # Hiển thị kết quả
        result_text = f"40km/h: {'YES' if is_40km_sign else 'NO'}"
        confidence_text = f"Confidence: {confidence:.2f}"
        time_text = f"Time: {inference_time*1000:.1f}ms"
        
        cv2.putText(image, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(image, confidence_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(image, time_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Hiển thị ảnh
        cv2.imshow(f"Traffic Sign Detection - {image_file}", image)
        key = cv2.waitKey(0)
        
        # Nhấn 'q' để thoát
        if key == ord('q'):
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = "../models/saved_models/traffic_sign_40km_model.tflite"
    image_dir = "../data/processed/test/40km"
    
    # Sử dụng hàm xử lý thư mục thay vì một ảnh đơn lẻ
    run_detection_on_directory(model_path, image_dir)
    
    # Nếu muốn chạy trên một ảnh cụ thể, bỏ comment dòng dưới và thêm đường dẫn ảnh
    # image_path = "../data/processed/test/40km/ten_file_anh.jpg"  # Thay bằng đường dẫn ảnh thực
    # run_detection_on_image(model_path, image_path)
