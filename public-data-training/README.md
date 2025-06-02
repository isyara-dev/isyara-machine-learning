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
├── BISINDO.zip                     # Bisindo Dataset   
├── README.md                       # Project documentation
├── notebook.ipynb                  # Jupyter Notebook version
├── notebook.py                     # Main script for training, evaluation & basic inference
└── requirements.txt                # Python dependencies
```

---

## 📦 Dataset

This project combines **2,158 images** from various BISINDO hand gesture datasets collected from the following sources

1. **Alfabet BISINDO - Achmad Noer**
   [https://www.kaggle.com/datasets/achmadnoer/alfabet-bisindo](https://www.kaggle.com/datasets/achmadnoer/alfabet-bisindo)

2. **BISINDO Dataset - Yunita Ayu**
   [https://www.kaggle.com/datasets/yunitayupratiwi/bisindo-dataset/data](https://www.kaggle.com/datasets/yunitayupratiwi/bisindo-dataset/data)

3. **BISINDO Letter Dataset - Alfredo**
   [https://www.kaggle.com/datasets/alfredolorentiars/bisindo-letter-dataset](https://www.kaggle.com/datasets/alfredolorentiars/bisindo-letter-dataset)

4. **BISINDO Hand Sign Detection Dataset - Rhio Sutoyo**
   [https://github.com/rhiosutoyo/Indonesian-Sign-Language-BISINDO-Hand-Sign-Detection-Dataset](https://github.com/rhiosutoyo/Indonesian-Sign-Language-BISINDO-Hand-Sign-Detection-Dataset)

Images are processed using **MediaPipe Hands** to extract 3D hand landmarks, which are then normalized and converted into coordinate-based features.

---

## 🚀 Project Pipeline

### 1. **Dataset Preparation**

* Alphabet gesture images (A-Z) organized by folder/class.
* Hand landmark features extracted from images using **MediaPipe Hands**.
* Data categorized into:

  * `one_hand` gestures
  * `two_hand` gestures

### 2. **Preprocessing & Oversampling**

* Normalized landmark features saved in CSV format.
* Oversampling applied to balance the number of samples across classes.

### 3. **Model Training**

* Two types of models are developed:

  * **1D CNN** (for sequential input)
  * **DNN** (for flattened feature vector input)

### 4. **Model Evaluation**

* CNN and DNN models are evaluated on test data.
* Evaluation includes accuracy, classification report, and confusion matrix.

### 5. **Model Conversion**

* Models are saved in both TensorFlow SavedModel and TensorFlow\.js formats.
* Ready for deployment in real-time web applications.

### 6. **Inference**

* Static inference can be done using `notebook.ipynb`, while real-time inference is supported on the web via `index.html`.
* `index.html` performs real-time gesture detection from webcam input.
* Uses `MediaPipe` for landmark extraction and `TensorFlow.js` for on-the-fly prediction in the browser.

---

## 📊 Model Evaluation

The models were evaluated using **accuracy**, **classification reports**, and **confusion matrices** on the test set.

### 🔹 DNN (Deep Neural Network)

* **Train Accuracy**: 99.30%
* **Test Accuracy**: 99.77%

The classification report shows near-perfect performance with consistent precision, recall, and f1-score across all classes. Only a few classes (e.g., ‘F’ and ‘P’) show slight drops in recall, but still remain within excellent performance range.

### 🔹 CNN 1D (1D Convolutional Neural Network)

* **Train Accuracy**: 99.77%
* **Test Accuracy**: 99.77%

The CNN 1D model also performs exceptionally well, with almost identical metrics to the DNN model. There is no significant variance across classes based on the confusion matrix.

### 📌 Summary:

Both models demonstrated outstanding ability to recognize hand gestures for A–Z letters on the test dataset. However, the unusually high performance may indicate potential **overfitting**, especially when considering the controlled dataset and data preparation process.

---

## ⚠️ Limitations & Considerations

* **Risk of Overfitting**:
  Oversampling was applied to balance the number of samples per class (up to 83 samples each). However, this process involved duplicating minority class data, which may have caused the model to memorize training data rather than generalize from it. This can result in overly optimistic test performance.

* **Synthetic Class Balance**:
  The balanced dataset was not achieved through diverse, naturally collected samples, but through duplication — particularly for classes with very few original samples (e.g., ‘N’ had only 28). This reduces data variability and limits the model’s exposure to realistic examples.

* **Need for Further Evaluation**:
  Additional evaluation using **real-world data** or **cross-validation** is needed to ensure the model's ability to generalize beyond the training dataset.

* **Real-Time Performance Uncertainty**:
  High test accuracy on static datasets does not guarantee the same performance in real-time applications. Variations in lighting, hand positioning, or camera angles may affect the model's ability to correctly classify hand gestures during actual usage.

---

## 🛠️ Setup & Model Execution

### On Google Colab

For model training & static inference:

1. Upload `BISINDO.zip` to your Google Drive
2. Set the dataset path in the `source` variable
3. Run `notebook.ipynb` sequentially in Colab
4. Output files: `label_encoder.pkl`, `saved_models/`, `tfjs_models/`

### Web Inference

#### Option 1: Live Demo
Access our ready-to-use demo:  
👉 [Open the app in your browser](https://isyara-inference.netlify.app/) (Recommended: use Google Chrome for best compatibility)

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

