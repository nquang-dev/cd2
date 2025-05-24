import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time

# Đường dẫn đến mô hình TFLite
MODEL_PATH = "/home/nquang/D_DAI-HOC/CD2/traffic_sign_detection/models/saved_models/traffic_sign_40km_model.tflite"

# Kích thước ảnh đầu vào của mô hình
INPUT_SIZE = (224, 224)

# Nhãn cho mô hình
LABELS = ["non40km", "40km"]

def load_tflite_model(model_path):
    # Load mô hình TFLite
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_image(image, input_size):
    # Resize ảnh về kích thước đầu vào của mô hình
    image = cv2.resize(image, input_size)
    # Chuyển sang định dạng RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Chuẩn hóa giá trị pixel về [0, 1]
    image = image / 255.0
    # Thêm batch dimension
    image = np.expand_dims(image, axis=0).astype(np.float32)
    return image

def predict(interpreter, image):
    # Lấy thông tin input và output của mô hình
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Đưa ảnh vào mô hình
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    # Lấy kết quả đầu ra (sigmoid output)
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

def main():
    # Load mô hình
    interpreter = load_tflite_model(MODEL_PATH)
    
    # Khởi tạo camera (0 là camera mặc định)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Không thể mở camera")
        return

    while True:
        # Ghi thời gian bắt đầu xử lý frame
        start_time = time.time()

        # Đọc frame từ camera
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc frame")
            break

        # Tiền xử lý ảnh
        processed_image = preprocess_image(frame, INPUT_SIZE)

        # Dự đoán
        predictions = predict(interpreter, processed_image)
        confidence = predictions[0][0]  # Sigmoid output
        predicted_class = 1 if confidence > 0.5 else 0  # Ngưỡng 0.5

        # Tính thời gian xử lý
        processing_time = time.time() - start_time

        # Hiển thị kết quả trên frame
        label = f"{LABELS[predicted_class]}: {confidence:.2f} (Time: {processing_time:.3f}s)"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Hiển thị frame
        cv2.imshow("Traffic Sign Detection", frame)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()