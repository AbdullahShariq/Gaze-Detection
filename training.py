import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Concatenate, Flatten, LeakyReLU, Lambda, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import re
from tqdm import tqdm

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Custom Attention Layer
class AttentionLayer(tf.keras.layers.Layer):
    def _init_(self, **kwargs):
        super(AttentionLayer, self)._init_(**kwargs)
    
    def build(self, input_shape):
        self.conv = Conv2D(1, kernel_size=1, use_bias=False)
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        attention = self.conv(inputs)
        attention = tf.nn.sigmoid(attention)  # Apply sigmoid within the layer
        attention_map = tf.broadcast_to(attention, tf.shape(inputs))  # Broadcast to match input shape
        return inputs * attention_map
    
    def compute_output_shape(self, input_shape):
        return input_shape

# Constants - Optimized
IMG_SIZE = (224, 224)  # Standard size for efficient models
ANGLE_RANGE = 30.0
BATCH_SIZE = 32  # Increased for faster training
WEIGHT_DECAY = 2e-5  # Slightly increased regularization
INIT_LR = 1e-4  # Higher initial learning rate for faster convergence

# Paths
DATASET_PATH = "/kaggle/input/columbia-dataset/columbia_gaze_data_set/"
PREPROCESSED_DIR = "/kaggle/working/preprocessed_data/"
MODEL_PATH = "/kaggle/working/gaze_detection_model.keras"
LOG_DIR = "/kaggle/working/logs/"
WEIGHTS_PATH = os.path.join(LOG_DIR, "best_weights.h5")

PREPROCESSED_IMAGES_PATH = "/kaggle/input/images/images.npy"
PREPROCESSED_ANGLES_PATH = "/kaggle/input/gaze-angles/gaze_angles.npy"

# Create necessary directories
os.makedirs(PREPROCESSED_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Initialize face detector
try:
    import dlib
    face_detector = dlib.get_frontal_face_detector()
    print("Using dlib for face detection")
    DLIB_AVAILABLE = True
except ImportError:
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    print("Using OpenCV Haar cascades for face detection")
    DLIB_AVAILABLE = False

def load_and_preprocess_data():
    """Load preprocessed data with memory-efficient operations"""
    # Check if preprocessed data exists at the specified paths
    if os.path.exists(PREPROCESSED_IMAGES_PATH) and os.path.exists(PREPROCESSED_ANGLES_PATH):
        print(f"Loading preprocessed data from {PREPROCESSED_IMAGES_PATH} and {PREPROCESSED_ANGLES_PATH}")
        
        # Memory-mapped loading for large arrays
        images = np.load(PREPROCESSED_IMAGES_PATH, mmap_mode='r')
        gaze_angles = np.load(PREPROCESSED_ANGLES_PATH)
        
        # Get actual shape info
        img_shape = images.shape[1:3]
        print(f"Loaded images shape: {images.shape}")
        
        # If image size doesn't match our target size, we need to resize
        if img_shape != IMG_SIZE:
            print(f"Image size mismatch: Expected {IMG_SIZE}, loaded {img_shape}. Resizing on-the-fly.")
            # We'll resize during batch training instead of all at once to save memory
            
        print(f"Loaded dataset: {images.shape} images, {gaze_angles.shape} labels")
        return images, gaze_angles
    
    # If preprocessed data is not found
    print("Preprocessed data not found at the specified paths.")
    print(f"Expected images at: {PREPROCESSED_IMAGES_PATH}")
    print(f"Expected gaze angles at: {PREPROCESSED_ANGLES_PATH}")
    
    # Placeholder for preprocessed data
    return np.array([]), np.array([])

class GazeErrorCallback(tf.keras.callbacks.Callback):
    """Custom callback to track gaze estimation error metrics"""
    def _init_(self, X_val, y_val, angle_range, resize_needed=False):
        super()._init_()
        self.X_val = X_val
        self.y_val = y_val
        self.angle_range = angle_range
        self.resize_needed = resize_needed
        self.history = {'yaw_mae': [], 'pitch_mae': [], 'angular_mae': [], 'epochs': []}
        self.best_angular_mae = float('inf')
        self.no_improvement_count = 0
    
    def on_epoch_end(self, epoch, logs=None):
        # Efficient validation with batched prediction to avoid memory issues
        batch_size = 64
        num_samples = len(self.X_val)
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        y_pred_list = []
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            
            X_batch = self.X_val[start_idx:end_idx]
            
            # Resize if needed
            if self.resize_needed:
                X_batch_resized = np.zeros((len(X_batch), IMG_SIZE[0], IMG_SIZE[1], X_batch.shape[-1]), dtype=X_batch.dtype)
                for j in range(len(X_batch)):
                    X_batch_resized[j] = cv2.resize(X_batch[j], IMG_SIZE)
                X_batch = X_batch_resized
            
            # Predict batch
            y_pred_batch = self.model.predict(X_batch, verbose=0)
            y_pred_list.append(y_pred_batch)
        
        # Combine predictions
        y_pred = np.vstack(y_pred_list)
        
        # Denormalize predictions and ground truth
        y_pred_denorm = y_pred * self.angle_range
        y_val_denorm = self.y_val * self.angle_range
        
        # Calculate metrics
        yaw_mae = np.mean(np.abs(y_val_denorm[:, 0] - y_pred_denorm[:, 0]))
        pitch_mae = np.mean(np.abs(y_val_denorm[:, 1] - y_pred_denorm[:, 1]))
        angular_errors = np.sqrt((y_val_denorm[:, 0] - y_pred_denorm[:, 0])**2 + 
                                (y_val_denorm[:, 1] - y_pred_denorm[:, 1])**2)
        angular_mae = np.mean(angular_errors)
        
        # Track history
        self.history['epochs'].append(epoch + 1)
        self.history['yaw_mae'].append(yaw_mae)
        self.history['pitch_mae'].append(pitch_mae)
        self.history['angular_mae'].append(angular_mae)
        
        # Check for improvement
        improved = False
        if angular_mae < self.best_angular_mae:
            self.best_angular_mae = angular_mae
            improved = True
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
        
        print(f"Epoch {epoch + 1} - Val Yaw MAE: {yaw_mae:.2f}°, Val Pitch MAE: {pitch_mae:.2f}°, Val Angular MAE: {angular_mae:.2f}°")
        print(f"Best Angular MAE: {self.best_angular_mae:.2f}°, Improved: {improved}, No improvement count: {self.no_improvement_count}")
    
    def plot_history(self):
        plt.figure(figsize=(12, 8))
        plt.plot(self.history['epochs'], self.history['yaw_mae'], label='Yaw MAE (°)')
        plt.plot(self.history['epochs'], self.history['pitch_mae'], label='Pitch MAE (°)')
        plt.plot(self.history['epochs'], self.history['angular_mae'], label='Angular MAE (°)')
        plt.xlabel('Epoch')
        plt.ylabel('Error (degrees)')
        plt.title('Validation Error Metrics')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(LOG_DIR, "error_metrics.png"))
        plt.close()
        print(f"Error metrics plot saved to {os.path.join(LOG_DIR, 'error_metrics.png')}")

def build_optimized_model():
    """Build an efficient model for gaze estimation with attention mechanism"""
    
    # Load EfficientNetV2S with transfer learning - more efficient than B0
    base_model = EfficientNetV2S(
        weights=None,  # Use local weights instead of downloading
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )

    base_model.load_weights('/kaggle/input/efficient-v2/keras/default/1/efficientnetv2-s_notop.h5')
    
    # Freeze early layers 
    for layer in base_model.layers[:-30]:  # Fine-tune only the last 30 layers
        layer.trainable = False
        
    # Build custom model
    inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    
    # Preprocessing 
    x = Lambda(lambda img: tf.keras.applications.efficientnet_v2.preprocess_input(img * 255.0))(inputs)
    
    # Pass through base model
    x = base_model(x)
    
    # Apply custom attention layer
    x = AttentionLayer()(x)
    
    # Global pooling to reduce parameters
    x = GlobalAveragePooling2D()(x)
    
    # Joint feature layers with regularization
    x = Dense(512, kernel_regularizer=l2(WEIGHT_DECAY))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Dropout(0.3)(x)
    
    # Split into two specialized branches for yaw and pitch
    
    # Yaw branch - optimized for horizontal movement detection
    yaw_branch = Dense(256, kernel_regularizer=l2(WEIGHT_DECAY))(x)
    yaw_branch = BatchNormalization()(yaw_branch)
    yaw_branch = LeakyReLU(0.1)(yaw_branch)
    yaw_branch = Dropout(0.2)(yaw_branch)
    yaw_output = Dense(1, name='yaw_output', activation='tanh')(yaw_branch)  # tanh constrains output to [-1, 1]
    
    # Pitch branch - optimized for vertical movement with more capacity
    pitch_branch = Dense(384, kernel_regularizer=l2(WEIGHT_DECAY))(x)  # More neurons for pitch
    pitch_branch = BatchNormalization()(pitch_branch)
    pitch_branch = LeakyReLU(0.1)(pitch_branch)
    pitch_branch = Dropout(0.2)(pitch_branch)
    pitch_branch = Dense(128, kernel_regularizer=l2(WEIGHT_DECAY))(pitch_branch)
    pitch_branch = BatchNormalization()(pitch_branch)
    pitch_branch = LeakyReLU(0.1)(pitch_branch)
    pitch_output = Dense(1, name='pitch_output', activation='tanh')(pitch_branch)  # tanh constrains output to [-1, 1]
    
    # Combine outputs
    outputs = Concatenate(name='combined_output')([yaw_output, pitch_output])
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile with AdamW optimizer (Adam with decoupled weight decay)
    optimizer = Adam(
        learning_rate=INIT_LR,
        clipnorm=1.0,  # Gradient clipping
        beta_1=0.9,    # Momentum parameter
        beta_2=0.999   # RMSprop parameter
    )
    
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    return model

def create_efficient_data_generator(resize_needed=False):
    """Create an efficient data generator for on-the-fly augmentation"""
    
    def preprocess_batch(img_batch):
        # Apply slight noise and contrast adjustment
        noise = np.random.normal(0, 0.01, img_batch.shape)
        img_batch = np.clip(img_batch + noise, 0, 1)
        
        # Random brightness adjustment
        brightness_factor = np.random.uniform(0.85, 1.15, size=(img_batch.shape[0], 1, 1, 1))
        img_batch = np.clip(img_batch * brightness_factor, 0, 1)
        
        return img_batch
    
    datagen = ImageDataGenerator(
        rotation_range=15,       # Moderate rotation
        width_shift_range=0.1,   # Horizontal shift
        height_shift_range=0.1,  # Vertical shift
        zoom_range=0.1,          # Zoom variation
        horizontal_flip=False,   # Gaze direction matters, don't flip
        fill_mode='nearest',
        preprocessing_function=preprocess_batch
    )
    
    return datagen

def train_model_with_efficient_scheduling(model, X_train, y_train, X_val, y_val, batch_size, epochs=40, resize_needed=False):
    """Train the model with cosine annealing learning rate schedule for better convergence"""
    
    # Create learning rate scheduler with cosine decay
    total_steps = int(len(X_train) / batch_size) * epochs
    warmup_steps = int(total_steps * 0.1)  # 10% warmup
    
    # Learning rate schedule with warmup and cosine decay
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=INIT_LR,
        decay_steps=total_steps - warmup_steps,
        alpha=1e-6  # Minimum learning rate
    )
    
    # Adjust optimizer's learning rate
    model.optimizer.learning_rate = lr_schedule
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,  # More patience to allow learning rate schedule to work
        restore_best_weights=True,
        verbose=1
    )
    
    checkpoint = ModelCheckpoint(
        os.path.join(LOG_DIR, "best_model.keras"),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # Custom callback to track gaze error metrics
    gaze_error_callback = GazeErrorCallback(X_val, y_val, ANGLE_RANGE, resize_needed)
    
    # Custom data generator for on-the-fly resizing if needed
    if resize_needed:
        train_generator = create_efficient_data_generator(resize_needed)
        
        # Custom generator function that resizes on-the-fly
        def generate_batches(X, y, batch_size):
            indices = np.arange(len(X))
            batch_count = 0
            
            while True:
                np.random.shuffle(indices)
                for start_idx in range(0, len(indices), batch_size):
                    batch_idx = indices[start_idx:start_idx + batch_size]
                    X_batch = X[batch_idx]
                    
                    # Resize batch
                    X_batch_resized = np.zeros((len(X_batch), IMG_SIZE[0], IMG_SIZE[1], X_batch.shape[-1]), dtype=X_batch.dtype)
                    for i in range(len(X_batch)):
                        X_batch_resized[i] = cv2.resize(X_batch[i], IMG_SIZE)
                    
                    # Apply augmentation
                    if batch_count % 5 != 0:  # Apply augmentation to 80% of batches
                        for i in range(len(X_batch_resized)):
                            X_batch_resized[i] = train_generator.random_transform(X_batch_resized[i])
                    
                    batch_count += 1
                    yield X_batch_resized, y[batch_idx]
        
        # Create generators
        train_gen = generate_batches(X_train, y_train, batch_size)
        val_steps = int(np.ceil(len(X_val) / batch_size))
        
        # Prepare validation data with resizing
        X_val_resized = np.zeros((len(X_val), IMG_SIZE[0], IMG_SIZE[1], X_val.shape[-1]), dtype=X_val.dtype)
        for i in range(len(X_val)):
            X_val_resized[i] = cv2.resize(X_val[i], IMG_SIZE)
        
        # Train with generator
        history = model.fit(
            train_gen,
            steps_per_epoch=len(X_train) // batch_size,
            epochs=epochs,
            validation_data=(X_val_resized, y_val),
            callbacks=[early_stopping, checkpoint, gaze_error_callback],
            verbose=1
        )
    else:
        # Standard training without resizing
        train_generator = create_efficient_data_generator()
        
        # Train with data augmentation
        history = model.fit(
            train_generator.flow(X_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(X_train) // batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, checkpoint, gaze_error_callback],
            verbose=1
        )
    
    # Plot training results
    gaze_error_callback.plot_history()
    
    return history, model

def evaluate_and_visualize(model, X_test, y_test, resize_needed=False):
    """Evaluate model with detailed metrics and visualizations"""
    # Prepare test data
    if resize_needed:
        X_test_resized = np.zeros((len(X_test), IMG_SIZE[0], IMG_SIZE[1], X_test.shape[-1]), dtype=X_test.dtype)
        for i in range(len(X_test)):
            X_test_resized[i] = cv2.resize(X_test[i], IMG_SIZE)
        X_test = X_test_resized
    
    # Evaluate
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Loss (MSE): {test_loss:.4f}")
    print(f"Test MAE (normalized): {test_mae:.4f}")
    
    # Predict for detailed metrics
    y_pred = model.predict(X_test)
    y_pred_denorm = y_pred * ANGLE_RANGE
    y_test_denorm = y_test * ANGLE_RANGE
    
    # Calculate error metrics
    yaw_error = np.mean(np.abs(y_test_denorm[:, 0] - y_pred_denorm[:, 0]))
    pitch_error = np.mean(np.abs(y_test_denorm[:, 1] - y_pred_denorm[:, 1]))
    angular_errors = np.sqrt((y_test_denorm[:, 0] - y_pred_denorm[:, 0])**2 + 
                          (y_test_denorm[:, 1] - y_pred_denorm[:, 1])**2)
    angular_mae = np.mean(angular_errors)
    
    print(f"Test Mean Absolute Error (Yaw): {yaw_error:.2f} degrees")
    print(f"Test Mean Absolute Error (Pitch): {pitch_error:.2f} degrees")
    print(f"Test Mean Angular Error: {angular_mae:.2f} degrees")
    
    # Error distribution visualization
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(np.abs(y_test_denorm[:, 0] - y_pred_denorm[:, 0]), bins=20, alpha=0.7, color='blue')
    plt.axvline(yaw_error, color='red', linestyle='--', linewidth=2)
    plt.title(f'Yaw Error Distribution (MAE: {yaw_error:.2f}°)')
    plt.xlabel('Absolute Error (degrees)')
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    plt.hist(np.abs(y_test_denorm[:, 1] - y_pred_denorm[:, 1]), bins=20, alpha=0.7, color='green')
    plt.axvline(pitch_error, color='red', linestyle='--', linewidth=2)
    plt.title(f'Pitch Error Distribution (MAE: {pitch_error:.2f}°)')
    plt.xlabel('Absolute Error (degrees)')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, "error_distribution.png"))
    
    # Sample predictions visualization
    num_samples = 5
    indices = np.random.choice(len(X_test), size=num_samples, replace=False)
    
    plt.figure(figsize=(15, 4*num_samples))
    for i, idx in enumerate(indices):
        img = X_test[idx]
        true_yaw, true_pitch = y_test_denorm[idx]
        pred_yaw, pred_pitch = y_pred_denorm[idx]
        error = angular_errors[idx]
        
        plt.subplot(num_samples, 1, i+1)
        plt.imshow(img)
        plt.title(f"True: Y={true_yaw:.1f}°, P={true_pitch:.1f}° | Pred: Y={pred_yaw:.1f}°, P={pred_pitch:.1f}° | Error: {error:.2f}°")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, "test_predictions.png"))
    plt.close()
    
    return yaw_error, pitch_error, angular_mae

def main():
    # Configure GPU memory growth if available
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f"Found {len(gpus)} GPU(s). Setting memory growth.")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Memory growth set successfully")
        else:
            print("No GPU found. Running on CPU.")
    except Exception as e:
        print(f"Error during GPU setup: {e}")

    # Load preprocessed data
    X, y = load_and_preprocess_data()
    if len(X) == 0 or len(y) == 0:
        print("Error: No valid images were processed or found.")
        return
    
    print(f"Working with dataset: {X.shape} images, {y.shape} labels")
    
    # Check if resizing is needed
    resize_needed = X.shape[1:3] != IMG_SIZE
    if resize_needed:
        print(f"Resize needed: Images will be resized from {X.shape[1:3]} to {IMG_SIZE} during training")

    # Split data with stratification
    # For stratification with continuous values, we create bins
    y_bins = np.zeros(len(y))
    for i in range(len(y)):
        # Create bins based on angle combinations
        yaw_bin = np.digitize(y[i, 0], np.linspace(-1, 1, 5))
        pitch_bin = np.digitize(y[i, 1], np.linspace(-1, 1, 5))
        y_bins[i] = yaw_bin * 10 + pitch_bin
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y_bins)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Train set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")
    
    # Build optimized model
    print("\nBuilding optimized model...")
    model = build_optimized_model()
    
    # Train with efficient learning rate schedule
    print("\nTraining model with efficient scheduling...")
    history, trained_model = train_model_with_efficient_scheduling(
        model, X_train, y_train, X_val, y_val, 
        batch_size=BATCH_SIZE, epochs=40, resize_needed=resize_needed
    )
    
    # Evaluate and visualize
    print("\nEvaluating on test set...")
    yaw_error, pitch_error, angular_mae = evaluate_and_visualize(
        trained_model, X_test, y_test, resize_needed=resize_needed
    )
    
    # Save final model
    trained_model.save(MODEL_PATH)
    print(f"Model saved as '{MODEL_PATH}'")
    
    # Report final metrics
    print("\nFinal Performance Metrics:")
    print(f"Yaw Error: {yaw_error:.2f}° | Pitch Error: {pitch_error:.2f}° | Angular Error: {angular_mae:.2f}°")
    
    # Compare with baseline (from original output)
    baseline_yaw = 8.33
    baseline_pitch = 18.49
    baseline_angular = 21.26
    
    yaw_improvement = (baseline_yaw - yaw_error) / baseline_yaw * 100
    pitch_improvement = (baseline_pitch - pitch_error) / baseline_pitch * 100
    angular_improvement = (baseline_angular - angular_mae) / baseline_angular * 100
    
    print("\nImprovement Over Baseline:")
    print(f"Yaw Error: {yaw_improvement:.1f}% improvement")
    print(f"Pitch Error: {pitch_improvement:.1f}% improvement")
    print(f"Angular Error: {angular_improvement:.1f}% improvement")
    
    return trained_model

if __name__ == "__main__":
    main()