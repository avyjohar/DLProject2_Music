# Music Genre Classification Project

## 1. Overview
This project classifies songs into 10 genres (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, and rock) using audio feature extraction and a deep learning model built with TensorFlow/Keras. The data is typically taken from a GTZAN-like structure with separate folders for each genre.

---

## 2. Dataset Structure
- A folder named `genres/` containing 10 subfolders (each named after a genre).
- Each subfolder has `.au` files (30-second clips).

Example structure:


Project 2 
└── genres 
├── blues 
│ ├── blues.00000.au │ ├── blues.00001.au │ └── ... 
├── classical 
├── country 
├── disco 
├── hiphop 
├── jazz 
├── metal 
├── pop 
├── reggae 
└── rock


---

## 3. Dependencies

- **Python** 3.x  
- **Librosa** (for audio feature extraction)  
- **NumPy** / **Pandas** (array/data handling)  
- **scikit-learn** (label encoding, splitting data, confusion matrix)  
- **TensorFlow/Keras** (building & training the neural network)  
- **matplotlib** (for plotting)

---

## 4. Project Steps

### 4.1. Audio Feature Extraction
**Function:** \`extract_features_extended(file_path)\`
1. Loads the audio file with \`librosa.load\` at 22050 Hz sample rate for up to 30 seconds.
2. Extracts:
   - **MFCCs** (\`n_mfcc=13\`)
   - **Spectral Contrast**
   - **Chroma Features**
   - **Zero Crossing Rate**
3. Averages these features over time, returning a **(33, )** array.

### 4.2. Dataset Preparation
**Function:** \`prepare_dataset(data_path)\`
1. Iterates through each subfolder (genre) in \`genres/\`.
2. For each \`.au\` file, it calls \`extract_features_extended(...)\`.
3. Collects all features in a NumPy array \`features\` and corresponding labels in \`labels\`.

### 4.3. Label Encoding & Train-Test Split
1. Converts string labels (genre names) to integers using \`LabelEncoder\`.
2. Splits data into:
   - Training set: 80%
   - Validation set: 20%

### 4.4. Model Definition
A feedforward neural network (MLP) in Keras:
- **Layers**:
  1. \`Dense(512, activation='relu')\` + \`BatchNormalization\` + \`Dropout(0.4)\`
  2. \`Dense(256, activation='relu')\` + \`BatchNormalization\` + \`Dropout(0.4)\`
  3. \`Dense(10, activation='softmax')\`
- **Loss**: \`categorical_crossentropy\`
- **Optimizer**: \`Adam\`
- **Metrics**: \`accuracy\`

### 4.5. Training
1. Typically run for \`epochs=50\`.
2. **EarlyStopping** monitors validation loss with a patience of 5.
3. **ModelCheckpoint** saves the best model (\`best_model.h5\`) when validation loss improves.

### 4.6. Evaluation
- After training, we evaluate on the validation set to see metrics (loss, accuracy).
- We also generate a **Classification Report** (precision, recall, F1) and a **Confusion Matrix**.

**Example Validation Accuracy**: ~73%

**Example Classification Report**:
\`\`\`
              precision    recall  f1-score   support
       blues       0.76      0.95      0.84        20
   classical       0.87      1.00      0.93        13
     country       0.83      0.89      0.86        27
       disco       0.63      0.57      0.60        21
      hiphop       0.50      0.60      0.55        15
        jazz       0.78      0.64      0.70        22
       metal       1.00      0.96      0.98        25
         pop       0.67      0.62      0.64        13
      reggae       0.67      0.43      0.53        23
        rock       0.52      0.62      0.57        21

    accuracy                           0.73       200
\`\`\`

---

## 5. Predicting on a New File
Example function:
\`\`\`python
def predict_genre(file_path, model, label_encoder):
    features = extract_features_extended(file_path)
    if features is None:
        return "Error extracting features."
    features = features.reshape(1, -1)
    prediction = model.predict(features)
    predicted_genre = label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_genre[0]
\`\`\`
Call it like:
\`\`\`python
new_file = "C:/path/to/new_song.au"
genre = predict_genre(new_file, best_model, label_encoder)
print("Predicted Genre:", genre)
\`\`\`

---

## 6. Saving & Loading the Model
After training:
\`\`\`python
model.save("genre_classification_model.keras")
\`\`\`
To load:
\`\`\`python
from tensorflow.keras.models import load_model
saved_model = load_model("genre_classification_model.keras")
\`\`\`
Then you can **predict** with \`saved_model\`.

---

## 7. Future Improvements
1. **Use CNNs** on Mel Spectrograms for better time-frequency analysis.
2. **Data Augmentation** (time/pitch shifting, adding noise).
3. **Hyperparameter Tuning** (learning rate, batch size, dropout, etc.).
4. **Scaling** features with \`StandardScaler\`.

---

## 8. How to Run
1. **Clone** or **download** this repository.
2. **Install** dependencies:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`
3. **Set** \`data_path\` to the location of your \`genres/\` folder.
4. **Run** the script or notebook. The best model will be saved to \`best_model.h5\` or \`genre_classification_model.keras\`.
5. **Evaluate** results, generate confusion matrix, or predict on new audio files.


```python

```
