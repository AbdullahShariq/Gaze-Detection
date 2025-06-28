import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Concatenate, Dense, Dropout, BatchNormalization, LeakyReLU, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import load_model


IMG_SIZE = (224, 224)
ANGLE_RANGE = 30.0
MODEL_PATH = "gaze_detection_model.keras"
WEIGHT_DECAY = 2e-5

CONFIDENCE_THRESHOLD = 0.7  # Confidence threshold for detection
MAX_VALID_ANGLE = 25.0      # Maximum valid angle (degrees)

#  Custom Attention Layer 
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.conv = tf.keras.layers.Conv2D(1, kernel_size=1, use_bias=False)

    def build(self, input_shape):
        self.conv.build(input_shape)  # This line is critical to initialize variables
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        attention = self.conv(inputs)
        attention = tf.nn.sigmoid(attention)
        attention_map = tf.broadcast_to(attention, tf.shape(inputs))
        return inputs * attention_map

    def compute_output_shape(self, input_shape):
        return input_shape


#  Rebuild Model to Load Weights 
def build_model():
    base_model = EfficientNetV2S(
        weights=None,
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )

    inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    x = Lambda(lambda img: tf.keras.applications.efficientnet_v2.preprocess_input(img * 255.0),
               output_shape=lambda s: s)(inputs)

    x = base_model(x)
    x = AttentionLayer()(x)
    x = GlobalAveragePooling2D()(x)

    x = Dense(512, kernel_regularizer=l2(WEIGHT_DECAY))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Dropout(0.3)(x)

    # Yaw branch
    yaw_branch = Dense(256, kernel_regularizer=l2(WEIGHT_DECAY))(x)
    yaw_branch = BatchNormalization()(yaw_branch)
    yaw_branch = LeakyReLU(0.1)(yaw_branch)
    yaw_branch = Dropout(0.2)(yaw_branch)
    yaw_output = Dense(1, name='yaw_output', activation='tanh')(yaw_branch)

    # Pitch branch
    pitch_branch = Dense(384, kernel_regularizer=l2(WEIGHT_DECAY))(x)
    pitch_branch = BatchNormalization()(pitch_branch)
    pitch_branch = LeakyReLU(0.1)(pitch_branch)
    pitch_branch = Dropout(0.2)(pitch_branch)
    pitch_branch = Dense(128, kernel_regularizer=l2(WEIGHT_DECAY))(pitch_branch)
    pitch_branch = BatchNormalization()(pitch_branch)
    pitch_branch = LeakyReLU(0.1)(pitch_branch)
    pitch_output = Dense(1, name='pitch_output', activation='tanh')(pitch_branch)

    outputs = Concatenate(name='combined_output')([yaw_output, pitch_output])
    model = Model(inputs=inputs, outputs=outputs)
    return model

#  Enhanced Model with Confidence Output 
def build_enhanced_model(base_model):
    """Add a confidence output to the existing model"""
    # Get the existing model's feature output before the final prediction layers
    feature_output = base_model.layers[-4].output  # Using the output before the final branches
    
    # Add a confidence estimation branch
    confidence_branch = Dense(128, kernel_regularizer=l2(WEIGHT_DECAY))(feature_output)
    confidence_branch = BatchNormalization()(confidence_branch)
    confidence_branch = LeakyReLU(0.1)(confidence_branch)
    confidence_output = Dense(1, activation='sigmoid', name='confidence')(confidence_branch)
    
    # Create new model with the confidence output
    enhanced_model = Model(inputs=base_model.inputs, 
                          outputs=[base_model.outputs[0], confidence_output])
    
    # Copy weights from the base model
    for layer in base_model.layers:
        if layer.name in enhanced_model.layers:
            enhanced_model.get_layer(layer.name).set_weights(layer.get_weights())
            
    return enhanced_model

#  Load Model Weights 
print("Rebuilding model architecture...")
model = build_model()
model.build((None, 224, 224, 3))  # Required to initialize layers like AttentionLayer
print("Loading weights from:", MODEL_PATH)
model.load_weights(MODEL_PATH)
print("Model loaded successfully.")

#  Image Preprocessing & Prediction 
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, IMG_SIZE)
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0), image_rgb

def is_gaze_detected(yaw, pitch, model_output=None):
   
    # Method 1: Check if angles are within valid range (not at extremes)
    angle_valid = (abs(yaw) <= MAX_VALID_ANGLE and abs(pitch) <= MAX_VALID_ANGLE)
    
    # Method 2: Calculate confidence based on prediction stability
    # Higher confidence when angles are moderate (not extreme)
    yaw_confidence = 1.0 - (abs(yaw) / ANGLE_RANGE)
    pitch_confidence = 1.0 - (abs(pitch) / ANGLE_RANGE)
    combined_confidence = (yaw_confidence + pitch_confidence) / 2.0
    
    # Decision logic
    is_detected = angle_valid and (combined_confidence >= CONFIDENCE_THRESHOLD)
    

    if is_detected:
        message = f"Gaze DETECTED with {combined_confidence:.2f} confidence"
    else:
        if not angle_valid:
            message = "Gaze NOT DETECTED: Angles out of valid range"
        else:
            message = f"Gaze NOT DETECTED: Low confidence ({combined_confidence:.2f})"
    
    return is_detected, combined_confidence, message

def predict_gaze(image_path):
    input_tensor, original = preprocess_image(image_path)
    
    # Get raw model outputs
    raw_output = model.predict(input_tensor)[0]
    
    # Scale to angle range
    prediction = raw_output * ANGLE_RANGE
    yaw, pitch = prediction


    FORWARD_YAW_THRESHOLD = 5.0
    FORWARD_PITCH_THRESHOLD = 5.0
    is_looking_forward = abs(yaw) < FORWARD_YAW_THRESHOLD and abs(pitch) < FORWARD_PITCH_THRESHOLD
    look_status = "LOOKING AT CAMERA" if is_looking_forward else "NOT LOOKING AT CAMERA"

    # Check if gaze is detected
    detected, confidence, message = is_gaze_detected(yaw, pitch, raw_output)
    
    print(f"\nPrediction:")
    print(f"Yaw: {yaw:.2f}°")
    print(f"Pitch: {pitch:.2f}°")
    print(f"Detection Status: {message}")
    print(f"Looking Status: {look_status}")

    
    plt.figure(figsize=(8, 6))
    plt.imshow(original)
    
    
    title_color = 'green' if detected else 'red'
    
    plt.title(f"Gaze Direction\nYaw: {yaw:.2f}°, Pitch: {pitch:.2f}°\n{look_status}, {message}",
              color=title_color, fontweight='bold')

    plt.axis('off')
    plt.show()
    
    return detected, yaw, pitch, confidence

if __name__ == "__main__":
    test_image = "NL4.jpg"
    if not os.path.exists(test_image):
        print(f"Image not found: {test_image}")
    else:
        detected, yaw, pitch, confidence = predict_gaze(test_image)