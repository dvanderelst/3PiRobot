from Library import DataProcessor
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import datetime

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Check GPU availability
print("🔍 Checking for GPU availability...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU detected: {gpus}")
    try:
        # Enable memory growth to avoid allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("🚀 GPU acceleration enabled!")
    except RuntimeError as e:
        print(f"⚠️  GPU configuration error: {e}")
else:
    print("💻 No GPU detected, using CPU")

# Configuration
sessions = ['session03', 'session04', 'session06']
TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 15

print("📊 Loading data...")
dc = DataProcessor.DataCollection(sessions)
dc.load_views(radius_mm=4000, opening_deg=90, output_size=(256, 256))
dc.load_profiles(az_min=-90, az_max=90, az_steps=19)

views = dc.views
profiles = dc.profiles
sonar_distance = dc.get_field('sonar_package', 'corrected_distance')
sonar_iid = dc.get_field('sonar_package', 'corrected_iid')

print(f"📈 Data shapes:")
print(f"   Views: {views.shape}")
print(f"   Profiles: {profiles.shape}")
print(f"   Sonar Distance: {sonar_distance.shape}")
print(f"   Sonar IID: {sonar_iid.shape}")

# Data preprocessing
def preprocess_data(profiles, sonar_distance, sonar_iid):
    """Preprocess data for training"""
    # Normalize profiles
    profile_scaler = StandardScaler()
    profiles_normalized = profile_scaler.fit_transform(profiles)
    
    # Normalize targets
    distance_scaler = StandardScaler()
    iid_scaler = StandardScaler()
    
    sonar_distance_normalized = distance_scaler.fit_transform(sonar_distance.reshape(-1, 1)).flatten()
    sonar_iid_normalized = iid_scaler.fit_transform(sonar_iid.reshape(-1, 1)).flatten()
    
    return (
        profiles_normalized, sonar_distance_normalized, sonar_iid_normalized,
        profile_scaler, distance_scaler, iid_scaler
    )

# Split data into train and test sets
X_train, X_test, y_distance_train, y_distance_test, y_iid_train, y_iid_test = train_test_split(
    profiles, sonar_distance, sonar_iid, 
    test_size=TEST_SIZE, random_state=RANDOM_STATE
)

print(f"🎯 Training set size: {len(X_train)}")
print(f"🎯 Test set size: {len(X_test)}")

# Preprocess the data
X_train_norm, y_distance_train_norm, y_iid_train_norm, profile_scaler, distance_scaler, iid_scaler = preprocess_data(
    X_train, y_distance_train, y_iid_train
)

# For test data, only transform (don't fit)
X_test_norm = profile_scaler.transform(X_test)  # Use the profile scaler, not distance scaler

# Build the neural network model
def build_sonar_predictor(input_shape):
    """Build a neural network for sonar prediction"""
    
    # Input layer
    inputs = keras.Input(shape=input_shape, name='profiles_input')
    
    # Feature extraction layers
    x = layers.Dense(128, activation='relu', name='dense_1')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(64, activation='relu', name='dense_2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(32, activation='relu', name='dense_3')(x)
    x = layers.BatchNormalization()(x)
    
    # Output layers - two separate heads for distance and IID
    distance_output = layers.Dense(1, name='distance_output')(x)
    iid_output = layers.Dense(1, name='iid_output')(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=[distance_output, iid_output], name='sonar_predictor')
    
    return model

# Create the model
model = build_sonar_predictor(input_shape=(X_train.shape[1],))

print("🏗️  Model architecture:")
model.summary()

# Compile the model with appropriate loss functions and metrics
def custom_loss_weights(y_true, y_pred):
    """Custom loss weights to balance distance and IID prediction"""
    # Distance prediction is more important, so give it higher weight
    return {'distance_output': 0.7, 'iid_output': 0.3}

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss={
        'distance_output': 'mean_squared_error',
        'iid_output': 'mean_squared_error'
    },
    loss_weights=custom_loss_weights(None, None),
    metrics={
        'distance_output': ['mae', 'mse'],
        'iid_output': ['mae', 'mse']
    }
)

# Callbacks for training
def get_callbacks():
    """Get callbacks for model training"""
    callbacks_list = [
        # Early stopping to prevent overfitting
        callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=PATIENCE, 
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate on plateau
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=PATIENCE//2,
            min_lr=1e-6,
            verbose=1
        ),
        # Model checkpoint to save best model
        callbacks.ModelCheckpoint(
            'best_sonar_predictor.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        # TensorBoard for visualization
        callbacks.TensorBoard(
            log_dir='logs',
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
    ]
    return callbacks_list

# Prepare data for Keras
print("📦 Preparing data for training...")
train_data = {
    'profiles_input': X_train_norm
}
train_targets = {
    'distance_output': y_distance_train_norm,
    'iid_output': y_iid_train_norm
}

# Train the model
print("🚀 Starting training...")
history = model.fit(
    train_data,
    train_targets,
    validation_split=0.1,  # Use 10% of training data for validation
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=get_callbacks(),
    verbose=1
)

# Evaluate the model
def evaluate_model(model, X_test, y_distance_test, y_iid_test, profile_scaler, distance_scaler, iid_scaler):
    """Evaluate model performance on test data"""
    
    # Normalize test data
    X_test_norm = profile_scaler.transform(X_test)
    
    # Make predictions
    y_distance_pred_norm, y_iid_pred_norm = model.predict(X_test_norm)
    
    # Inverse transform predictions to original scale
    y_distance_pred = distance_scaler.inverse_transform(y_distance_pred_norm)
    y_iid_pred = iid_scaler.inverse_transform(y_iid_pred_norm)
    
    # Calculate metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    distance_mae = mean_absolute_error(y_distance_test, y_distance_pred)
    distance_mse = mean_squared_error(y_distance_test, y_distance_pred)
    distance_r2 = r2_score(y_distance_test, y_distance_pred)
    
    iid_mae = mean_absolute_error(y_iid_test, y_iid_pred)
    iid_mse = mean_squared_error(y_iid_test, y_iid_pred)
    iid_r2 = r2_score(y_iid_test, y_iid_pred)
    
    print("📊 Model Evaluation Results:")
    print(f"   Distance Prediction - MAE: {distance_mae:.4f}, MSE: {distance_mse:.4f}, R²: {distance_r2:.4f}")
    print(f"   IID Prediction - MAE: {iid_mae:.4f}, MSE: {iid_mse:.4f}, R²: {iid_r2:.4f}")
    
    return {
        'distance_mae': distance_mae,
        'distance_mse': distance_mse,
        'distance_r2': distance_r2,
        'iid_mae': iid_mae,
        'iid_mse': iid_mse,
        'iid_r2': iid_r2
    }

# Plot training history
def plot_training_history(history):
    """Plot training and validation metrics"""
    
    plt.figure(figsize=(15, 10))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot distance MAE
    plt.subplot(2, 2, 2)
    plt.plot(history.history['distance_output_mae'], label='Training Distance MAE')
    plt.plot(history.history['val_distance_output_mae'], label='Validation Distance MAE')
    plt.title('Distance MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    # Plot IID MAE
    plt.subplot(2, 2, 3)
    plt.plot(history.history['iid_output_mae'], label='Training IID MAE')
    plt.plot(history.history['val_iid_output_mae'], label='Validation IID MAE')
    plt.title('IID MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    # Plot learning rate (if available)
    plt.subplot(2, 2, 4)
    if 'lr' in history.history:
        plt.plot(history.history['lr'], label='Learning Rate')
    else:
        plt.plot([], label='Learning Rate (not tracked)')
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

# Save the model and scalers
def save_model_and_scalers(model, profile_scaler, distance_scaler, iid_scaler):
    """Save model and preprocessing scalers"""
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Generate timestamp for model naming
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = f'sonar_predictor_{timestamp}'
    
    # Save the model
    model.save(f'models/{model_name}.h5')
    print(f"💾 Model saved as: models/{model_name}.h5")
    
    # Save scalers using joblib
    import joblib
    joblib.dump(distance_scaler, f'models/{model_name}_distance_scaler.pkl')
    joblib.dump(iid_scaler, f'models/{model_name}_iid_scaler.pkl')
    joblib.dump(profile_scaler, f'models/{model_name}_profile_scaler.pkl')
    
    print(f"💾 Scalers saved with model prefix: {model_name}")
    
    return model_name

# Main execution
if __name__ == "__main__":
    print("🎯 Starting sonar prediction model training...")
    
    # Train and evaluate
    metrics = evaluate_model(model, X_test, y_distance_test, y_iid_test, profile_scaler, distance_scaler, iid_scaler)
    
    # Plot training history
    plot_training_history(history)
    
    # Save model and scalers
    model_name = save_model_and_scalers(model, profile_scaler, distance_scaler, iid_scaler)
    
    print("✅ Training complete!")
    print(f"📁 Model and artifacts saved with prefix: {model_name}")