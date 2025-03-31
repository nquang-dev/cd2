from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

def create_model(input_shape=(224, 224, 3), learning_rate=0.0001):
    """
    Tạo mô hình CNN dựa trên MobileNetV2 (transfer learning)
    """
    # Tải mô hình MobileNetV2 đã được pre-trained trên ImageNet
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Đóng băng các lớp của mô hình cơ sở
    for layer in base_model.layers:
        layer.trainable = False
    
    # Tạo mô hình mới
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Bài toán phân loại nhị phân: 40km/h vs không phải
    ])
    
    # Biên dịch mô hình
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', 'Precision', 'Recall']
    )
    
    return model

def create_custom_model(input_shape=(224, 224, 3)):
    """
    Tạo mô hình CNN tùy chỉnh nhỏ hơn (phù hợp với Raspberry Pi)
    """
    model = Sequential([
        # Block 1
        Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        
        # Block 2
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Block 3
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Classification block
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'Precision', 'Recall']
    )
    
    return model
