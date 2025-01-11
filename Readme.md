# Project: Music Genre Classification

## Overview
This project develops a machine learning model to classify audio tracks into different genres based on extracted features. The dataset contains audio files for genres such as blues, classical, country, and more.

## Data Source
- **Path**: `C:\Users\Aavneet Singh Johar\IIT_Kanpur_Edvancer\Python\Deep Learning projects\Project 2\genres`
- **Genres**: Blues, Classical, Country, Disco, Hiphop, Jazz, Metal, Pop, Reggae, Rock

## Process Overview
1. **Feature Extraction**:
   - Extract features such as MFCCs, spectral contrast, chroma features, and zero-crossing rate using Librosa.
   - Combine features into a single feature vector for each audio file.

2. **Data Preparation**:
   - Encode genre labels using `LabelEncoder`.
   - Split the data into training and validation sets (80-20 split).

3. **Model Architecture**:
   - A fully connected neural network with the following layers:
     - Input layer
     - Dense layers with ReLU activation
     - Batch Normalization and Dropout layers
     - Output layer with softmax activation

4. **Model Training**:
   - Use Adam optimizer and categorical crossentropy loss.
   - Incorporate early stopping and model checkpointing.

5. **Evaluation**:
   - Metrics include precision, recall, F1-score, and confusion matrix.

6. **Prediction**:
   - Predict genres for new audio files using the trained model.

## Results
- **Validation Accuracy**: 73%
- **Best Model**: Saved as `best_model.h5`.
- **Top Features**:
     - MFCCs
     - Spectral Contrast
     - Chroma Features

## How to Run the Project
1. **Requirements**:
   - Python 3.x
   - TensorFlow, NumPy, pandas, sklearn, librosa, and matplotlib.

2. **Steps**:
   - Extract features from audio files using `prepare_dataset()`.
   - Train the model using the provided script.
   - Evaluate the model on validation data.
   - Save and load the trained model for predictions.

## Example Usage
```python
# Predict genre for a new audio file
from tensorflow.keras.models import load_model

# Load saved model
model = load_model('best_model.h5')

# Extract features for the new audio file
new_audio_path = "path/to/new/audio/file.au"
features = extract_features_extended(new_audio_path)

# Predict genre
predicted_genre = label_encoder.inverse_transform([np.argmax(model.predict(features.reshape(1, -1)))])
print("Predicted Genre:", predicted_genre[0])
```

## Files
- **best_model.h5**: Trained model file.
- **genre_classification_model.keras**: Model in Keras recommended format.
- **Python script**: End-to-end script to preprocess data, train, and evaluate the model.

## Acknowledgments
- **Frameworks**: TensorFlow, Keras, and Librosa.
- **Dataset**: GTZAN Genre Dataset.

## License
This project is for educational purposes and is not intended for commercial use.

## Author
Aavneet Singh Johar



```python

```
