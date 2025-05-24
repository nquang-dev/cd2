import cv2
import numpy as np
import time
from picamera2 import Picamera2
import tflite_runtime.interpreter as tflite
import serial

# Đường dẫn đến mô hình TFLite
MODEL_PATH = "/home/pi/traffic_sign_detection/models/saved_models/traffic_sign_40km_model.tflite"

# Kích thước ảnh đầu vào của mô hình
INPUT_SIZE = (224, 224)

# Nhãn cho mô hình
LABELS = ["non40km", "40km"]

# Kết nối Serial với Arduino
ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)

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
    interpreter = load_tflite_model(MODEL_PATH)
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
    picam2.start()

    while True:
        start_time = time.time()
        frame = picam2.capture_array()
        processed_image = preprocess_image(frame, INPUT_SIZE)
        predictions = predict(interpreter, processed_image)
        confidence = predictions[0][0]
        predicted_class = 1 if confidence > 0.5 else 0
        processing_time = time.time() - start_time

        label = f"{LABELS[predicted_class]}: {confidence:.2f} (Time: {processing_time:.3f}s)"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Gửi lệnh tới Arduino
        if predicted_class == 1:  # Biển 40km/h
            ser.write(b'S')  # Giảm tốc độ
            print("Detected 40km/h sign, slowing down...")
        else:
            ser.write(b'F')  # Tốc độ bình thường
            print("No 40km/h sign, normal speed...")

        cv2.imshow("Traffic Sign Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    picam2.stop()
    cv2.destroyAllWindows()
    ser.close()

if __name__ == "__main__":
    main()