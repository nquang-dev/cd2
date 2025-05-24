import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import datetime
from model import create_model

def train_model(train_dir, val_dir, model_save_dir, checkpoint_dir, batch_size=32, epochs=50, img_size=(224, 224)):
    """
    Huấn luyện mô hình phát hiện biển báo 40km/h
    """
    # Tạo thư mục lưu mô hình và checkpoint
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        classes=['non40km', '40km']  #0=non40km, 1=40km
    )
    
    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        classes=['non40km', '40km']
    )
    
    # Tạo mô hình
    model = create_model(input_shape=(*img_size, 3))
    
    # Callbacks
    checkpoint_path = os.path.join(checkpoint_dir, 'model-{epoch:02d}-{val_accuracy:.4f}.h5')
    checkpoint = ModelCheckpoint(
        checkpoint_path, 
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    # Huấn luyện mô hình
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        callbacks=[checkpoint, early_stopping, reduce_lr, tensorboard]
    )
    
    # Lưu mô hình cuối cùng
    model.save(os.path.join(model_save_dir, 'traffic_sign_40km_model.h5'))
    
    # Lưu mô hình dạng TFLite (để triển khai trên Raspberry Pi)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    with open(os.path.join(model_save_dir, 'traffic_sign_40km_model.tflite'), 'wb') as f:
        f.write(tflite_model)
    
    print("Đã huấn luyện và lưu mô hình thành công!")
    return history

if __name__ == "__main__":
    train_dir = "../data/processed/train"
    val_dir = "../data/processed/val"
    model_save_dir = "../models/saved_models"
    checkpoint_dir = "../models/checkpoints"
    
    train_model(train_dir, val_dir, model_save_dir, checkpoint_dir)
