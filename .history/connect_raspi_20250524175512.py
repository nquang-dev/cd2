import cv2
import numpy as np
import time
from picamera2 import Picamera2
import tflite_runtime.interpreter as tflite

# Đường dẫn đến mô hình TFLite
MODEL_PATH = "/home/pi/traffic_sign_detection/models/saved_models/traffic_sign_40km_model.tflite"

# Kích thước ảnh đầu vào của mô hình
INPUT_SIZE = (224, 224)

# Nhãn cho mô hình
LABELS = ["non40km", "40km"]

def load_tflite_model(model_path):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_image(image, input_size):
    image = cv2.resize(image, input_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    image = np.expand_dims(image, axis=0).astype(np.float32)
    return image

def predict(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

def main():
    # Load mô hình
    interpreter = load_tflite_model(MODEL_PATH)
    
    # Khởi tạo Picamera2
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
    picam2.start()

    while True:
        start_time = time.time()

        # Chụp ảnh từ Picamera2
        frame = picam2.capture_array()

        # Tiền xử lý ảnh
        processed_image = preprocess_image(frame, INPUT_SIZE)

        # Dự đoán
        predictions = predict(interpreter, processed_image)
        confidence = predictions[0][0]
        predicted_class = 1 if confidence > 0.5 else 0

        # Tính thời gian xử lý
        processing_time = time.time() - start_time

        # Hiển thị kết quả trên frame
        label = f"{LABELS[predicted_class]}: {confidence:.2f} (Time: {processing_time:.3f}s)"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Hiển thị frame
        cv2.imshow("Traffic Sign Detection", frame)

        # Gửi tín hiệu điều khiển tốc độ (sẽ thêm ở Bước 3)
        if predicted_class == 1:  # Biển 40km/h
            print("Detected 40km/h sign, slowing down...")
            # TODO: Gửi lệnh giảm tốc độ
        else:
            print("No 40km/h sign, normal speed...")
            # TODO: Gửi lệnh tốc độ bình thường

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng tài nguyên
    picam2.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()