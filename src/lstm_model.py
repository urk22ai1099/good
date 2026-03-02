"""
AgroMind+ LSTM Model
Advanced LSTM with Attention Mechanism for Crop Prediction
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class AttentionLayer(layers.Layer):
    """Custom Attention Layer for LSTM"""
    
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        # Calculate attention scores
        score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # Apply attention weights
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector

class AgroMindLSTM:
    def __init__(self, sequence_length=4, n_features=9):
        """
        Initialize LSTM model for crop prediction
        
        Parameters:
        - sequence_length: Number of time steps (weeks)
        - n_features: Number of input features
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.history = None
        
        # Crop information
        self.crops = ['Aman_Rice', 'Boro_Rice', 'Wheat', 'Maize', 'Millets', 'Pulses', 'Cotton']
    
    def prepare_data(self, sequences, labels):
        """Prepare data for LSTM training"""
        print("\n" + "="*80)
        print("Preparing Data for LSTM Training")
        print("="*80)
        
        # Reshape sequences for scaling
        n_samples = sequences.shape[0]
        sequences_2d = sequences.reshape(-1, self.n_features)
        
        # Scale features
        sequences_scaled = self.scaler.fit_transform(sequences_2d)
        sequences_scaled = sequences_scaled.reshape(n_samples, self.sequence_length, self.n_features)
        
        # Encode labels
        labels_encoded = self.label_encoder.fit_transform(labels)
        labels_categorical = keras.utils.to_categorical(labels_encoded)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            sequences_scaled, labels_categorical,
            test_size=0.2, random_state=42, stratify=labels_encoded
        )
        
        print(f"✓ Training set: {X_train.shape}")
        print(f"✓ Testing set: {X_test.shape}")
        print(f"✓ Number of classes: {labels_categorical.shape[1]}")
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self, n_classes):
        """Build LSTM model with attention mechanism"""
        print("\n" + "="*80)
        print("Building LSTM Model with Attention Mechanism")
        print("="*80)
        
        model = models.Sequential([
            # First LSTM layer with return sequences
            layers.LSTM(
                128,
                input_shape=(self.sequence_length, self.n_features),
                return_sequences=True,
                dropout=0.2,
                recurrent_dropout=0.2
            ),
            layers.BatchNormalization(),
            
            # Second LSTM layer with return sequences for attention
            layers.LSTM(
                64,
                return_sequences=True,
                dropout=0.2,
                recurrent_dropout=0.2
            ),
            layers.BatchNormalization(),
            
            # Attention layer
            AttentionLayer(),
            
            # Dense layers
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(n_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
        )
        
        self.model = model
        
        print("\n" + "="*80)
        print("Model Architecture:")
        print("="*80)
        model.summary()
        
        return model
    
    def train(self, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
        """Train the LSTM model"""
        print("\n" + "="*80)
        print("Training LSTM Model")
        print("="*80)
        
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        
        model_checkpoint = callbacks.ModelCheckpoint(
            '../models/best_lstm_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr, model_checkpoint],
            verbose=1
        )
        
        print("\n✓ Training completed")
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        print("\n" + "="*80)
        print("Model Evaluation")
        print("="*80)
        
        # Predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Accuracy
        accuracy = accuracy_score(y_true, y_pred)
        print(f"\n✓ Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Top-3 accuracy
        top3_acc = np.mean([y_true[i] in np.argsort(y_pred_proba[i])[-3:] 
                            for i in range(len(y_true))])
        print(f"✓ Top-3 Accuracy: {top3_acc:.4f} ({top3_acc*100:.2f}%)")
        
        # Classification report
        print("\n" + "="*80)
        print("Classification Report:")
        print("="*80)
        crop_names = self.label_encoder.classes_
        print(classification_report(y_true, y_pred, target_names=crop_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        return accuracy, top3_acc, cm, y_pred_proba
    
    def plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        axes[0].plot(self.history.history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[0].plot(self.history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss plot
        axes[1].plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        axes[1].plot(self.history.history['val_loss'], label='Val Loss', linewidth=2)
        axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../models/training_history.png', dpi=300, bbox_inches='tight')
        print("\n✓ Training history plot saved to: ../models/training_history.png")
        plt.close()
    
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix - LSTM Crop Prediction', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Crop')
        plt.ylabel('True Crop')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('../models/confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("✓ Confusion matrix saved to: ../models/confusion_matrix.png")
        plt.close()
    
    def predict_top_crops(self, sequence, top_k=4):
        """Predict top-k crops for a given sequence"""
        # Scale input
        sequence_2d = sequence.reshape(-1, self.n_features)
        sequence_scaled = self.scaler.transform(sequence_2d)
        sequence_scaled = sequence_scaled.reshape(1, self.sequence_length, self.n_features)
        
        # Predict
        predictions = self.model.predict(sequence_scaled, verbose=0)[0]
        
        # Get top-k predictions
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        top_crops = self.label_encoder.inverse_transform(top_indices)
        top_probabilities = predictions[top_indices]
        
        results = []
        for i, (crop, prob) in enumerate(zip(top_crops, top_probabilities)):
            results.append({
                'rank': i + 1,
                'crop': crop,
                'suitability': prob,
                'confidence': f"{prob*100:.2f}%"
            })
        
        return results
    
    def save_model(self):
        """Save complete model and preprocessors"""
        print("\n" + "="*80)
        print("Saving Model and Preprocessors")
        print("="*80)
        
        # Save Keras model
        self.model.save('../models/agromind_lstm_model.h5')
        print("✓ LSTM model saved to: ../models/agromind_lstm_model.h5")
        
        # Save preprocessors
        joblib.dump(self.scaler, '../models/feature_scaler.pkl')
        print("✓ Feature scaler saved")
        
        joblib.dump(self.label_encoder, '../models/label_encoder.pkl')
        print("✓ Label encoder saved")
        
        # Save model metadata
        metadata = {
            'crops': list(self.label_encoder.classes_),
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'feature_names': ['N', 'P', 'K', 'pH', 'Temperature', 'Humidity', 
                             'Moisture', 'Rainfall', 'Sunlight']
        }
        joblib.dump(metadata, '../models/model_metadata.pkl')
        print("✓ Model metadata saved")
    
    @classmethod
    def load_model(cls):
        """Load pre-trained model"""
        instance = cls()
        instance.model = keras.models.load_model(
            '../models/agromind_lstm_model.h5',
            custom_objects={'AttentionLayer': AttentionLayer}
        )
        instance.scaler = joblib.load('../models/feature_scaler.pkl')
        instance.label_encoder = joblib.load('../models/label_encoder.pkl')
        return instance

def main():
    """Main training pipeline"""
    print("="*80)
    print("AgroMind+ LSTM Training Pipeline")
    print("="*80)
    
    # Load sequences
    print("\nLoading data...")
    sequences = np.load('../data/sequences.npy')
    labels = np.load('../data/labels.npy')
    print(f"✓ Loaded {len(sequences)} sequences")
    
    # Initialize model
    agromind = AgroMindLSTM(sequence_length=4, n_features=9)
    
    # Prepare data
    X_train, X_test, y_train, y_test = agromind.prepare_data(sequences, labels)
    
    # Build model
    n_classes = y_train.shape[1]
    agromind.build_model(n_classes)
    
    # Train model
    agromind.train(X_train, y_train, X_test, y_test, epochs=100, batch_size=32)
    
    # Evaluate model
    accuracy, top3_acc, cm, predictions = agromind.evaluate(X_test, y_test)
    
    # Plot results
    agromind.plot_training_history()
    agromind.plot_confusion_matrix(cm)
    
    # Save model
    agromind.save_model()
    
    # Demo prediction
    print("\n" + "="*80)
    print("Demo: Top-4 Crop Predictions")
    print("="*80)
    
    sample_sequence = X_test[0]
    # Inverse transform to original scale for display
    sample_2d = sample_sequence.reshape(-1, 9)
    sample_original = agromind.scaler.inverse_transform(sample_2d)
    sample_original = sample_original.reshape(4, 9)
    
    results = agromind.predict_top_crops(sample_original, top_k=4)
    
    print("\nTop-4 Recommended Crops:")
    for result in results:
        print(f"  {result['rank']}. {result['crop']:<15} - Suitability: {result['confidence']}")
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"\nFinal Results:")
    print(f"  - Test Accuracy: {accuracy*100:.2f}%")
    print(f"  - Top-3 Accuracy: {top3_acc*100:.2f}%")
    print(f"\nAll files saved to: ../models/")

if __name__ == "__main__":
    main()