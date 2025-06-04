# ISYARA: Indonesian Sign Language Recognition 

This project is a **BISINDO (Indonesian Sign Language)** recognition system based on **hand landmark coordinate extraction** using **MediaPipe Hands**. The extracted features are trained using **Machine Learning models (DNN & 1D CNN)** to classify hand gestures representing letters A–Z. The project also supports **real-time web-based inference** using **TensorFlow\.js**.

---

## ✨ ISYARA Machine Learning Team

* **Davin Ghani Ananta Kusuma** - MC299D5Y1599
* **Nauval Gymnasti** - MC299D5Y1716
* **Rizka Alfadilla** - MC299D5Y1776

---

## 📁 Project Structure

```bash
├── data-collection/
│   ├── collection.py               # data collection program
│   └── requirements.txt            # Python dependencies for data collection
│     
├── model/
│   ├── cnn_tfjs_model.zip          # CNN model in TensorFlow.js format
│   ├── dnn_tfjs_model.zip          # DNN model in TensorFlow.js format
│   ├── label_encoder.pkl           # Label encoder for gesture classification 
│   ├── saved_models_cnn.zip        # CNN model (TensorFlow SavedModel format) 
│   └── saved_models_dnn.zip        # DNN model (TensorFlow SavedModel format)
│
├── web-inference/
│   ├── group1-shard1of1.bin        # TFJS model weights 
│   ├── index.html                  # Real-time inference web page 
│   └── model.json                  # TFJS model architecture
│
├── BISINDO.csv                     # Bisindo Self Collected Dataset   
├── README.md                       # Project documentation
├── notebook.ipynb                  # Jupyter Notebook version
├── notebook.py                     # Main script for training, evaluation & basic inference
└── requirements.txt                # Python dependencies
```

---

## 📦 Dataset

This project uses a **self-collected dataset of 7,800 data** generated using the custom data collection tool in the `data-collection` folder. The dataset consists of BISINDO hand gesture data for letters A–Z, captured and processed directly using the provided script.

### Data Collection Tool

Hand gesture data are collected and processed using the `collection.py` script, which utilizes **MediaPipe Hands** to extract 3D hand landmarks. These landmarks are normalized and saved as coordinate-based features in CSV format.

#### How to Collect Data

1. Open a terminal in the `data-collection` directory and run `pip install -r requirements.txt`. (It is recommended to use a virtual environment)
2. Run the data collection script `python collection.py`.
3. Enter the gesture label (e.g., A, B, C).
4. Enter the number of samples to collect.
5. Choose the mode: `1` for one hand, `2` for two hands.
6. Follow the on-screen instructions to capture hand gesture data using your webcam.
7. Collected data will be saved as `one_hand_landmarks.csv` or `two_hand_landmarks.csv` in the same directory.

Images are processed in real-time using **MediaPipe Hands** to extract and normalize 3D hand landmarks, which are then stored as features for model training.

---

## 🚀 Project Pipeline

### 1. **Data Collection (not in notebook)**

* Hand gesture data for letters A–Z is collected using the `collection.py` tool in the `data-collection` folder.
* Each sample is captured via webcam, and 3D hand landmarks are extracted in real-time using **MediaPipe Hands**.
* Data is categorized into:
  * `one_hand` gestures (saved in `one_hand_landmarks.csv`)
  * `two_hand` gestures (saved in `two_hand_landmarks.csv`)
* All collected and normalized landmark features are merged and saved as `BISINDO.csv`.

### 2. **Data Splitting**
* The collected dataset (`BISINDO.csv`) is split into training and test sets to evaluate model performance.
* Typically, **80%** of the data is used for training and **20%** for testing.
* The split is performed **stratified by class** to ensure each letter (A–Z) is proportionally represented in both sets.
* This helps provide a reliable estimate of how well the model will generalize to unseen data.

### 3. **Model Training**

* Two types of models are developed:
  * **1D CNN** (for sequential landmark input)
  * **DNN** (for flattened feature vector input)
* Training and evaluation are performed using `notebook.ipynb` or `notebook.py`.

### 4. **Model Evaluation**

* Both CNN and DNN models are evaluated on a held-out test set.
* Evaluation metrics include accuracy, classification report, and confusion matrix.

### 5. **Model Conversion**

* Trained models are exported in both TensorFlow SavedModel and TensorFlow.js formats.
* This enables easy deployment for both Python and web-based inference.

### 6. **Inference**

* **Static inference**: Use `notebook.ipynb` to test the model on images or CSV data.
* **Real-time inference**: Open `web-inference/index.html` in your browser for live webcam gesture recognition.
  * The web app uses **MediaPipe** for landmark extraction and **TensorFlow.js** for real-time prediction.

---

## 📊 Model Evaluation

The models were evaluated using **accuracy**, **classification reports**, and **confusion matrices** on the test set.

### 🔹 CNN 1D (1D Convolutional Neural Network)

* **Train Accuracy**: 99.64%
* **Test Accuracy**: 99.04%

The CNN 1D model also performs exceptionally well, with almost identical metrics to the DNN model. There is no significant variance across classes based on the confusion matrix.

### 🔹 DNN (Deep Neural Network)

* **Train Accuracy**: 99.46%
* **Test Accuracy**: 98.97%

The classification report shows near-perfect performance with consistent precision, recall, and f1-score across all classes. Only a few classes (e.g., ‘L’, ‘M’ and ‘N’) show slight drops in recall, but still remain within excellent performance range.

---

## 📌 Summary:

Both the CNN and DNN models achieved **excellent performance** on the newly collected dataset containing **7,800 hand landmark coordinate samples**, with test accuracies of **99.04% (CNN)** and **98.97% (DNN)**. The models successfully learned to classify BISINDO hand gestures (A–Z) using these coordinate-based features with high consistency.

Compared to the earlier models trained on public datasets, the slightly lower test accuracy in this retraining indicates **better generalization**, as the new dataset reflects more **natural variations in hand movement and positioning**.

---

## ⚠️ Limitations & Considerations

* **Limited User Diversity**:
  The landmark dataset, while larger, was collected from only three individuals. This may limit the generalization of the model to hands with different shapes, sizes, or movement styles.

* **Absence of Cross-User Validation**:
  Although the dataset was split for training and testing, all data still come from the same group. **Cross-user validation** (e.g., leave-one-user-out) is important to test how well the model generalizes to new users.

* **Real-Time Application Uncertainty**:
  High performance on static coordinate data does not always translate to stable real-time inference. Factors like tracking jitter, missed landmarks, or rapid hand motion in live webcam feeds may affect accuracy.

* **Noise Sensitivity**:
  Landmark coordinates from MediaPipe may vary slightly due to hand tremors, occlusion, or camera lag. The model's robustness to such **landmark noise** should be evaluated further, especially in edge cases.

---

## 🛠️ Setup & Model Execution

### On Google Colab

For model training & static inference:

1. Upload `BISINDO.csv` to your Google Drive
2. Set the dataset path in the `df` variable
3. Run `notebook.ipynb` sequentially in Colab
4. Output files: `label_encoder.pkl`, `saved_models/`, `tfjs_models/`

### Web Inference

#### Option 1: Live Demo
Access our ready-to-use demo:  
👉 [Open the app in your browser](https://isyara-inference-collected.netlify.app/) (Recommended: use Google Chrome for best compatibility)

#### Option 2: Local Deployment
To run locally with the latest model:
1. Download `index.html` and the corresponding `_tfjs_model.zip` file
2. Extract `_tfjs_model.zip`
3. Ensure `model.json` and `*.bin` files are in the same directory
4. Open `index.html` in a modern browser (Chrome recommended)

---

## 💡 Usage Notes

* Ensure good lighting conditions during inference.
* Hand gestures should be clearly visible and within the webcam frame.
* If only one hand is detected, the system will still work using a fallback approach.

---
