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
├── collected-data-training/
│   ├── data-collection/        # Scripts and tools for collecting hand gesture data
│   ├── model/                  # Trained machine learning models
│   ├── web-inference/          # Web app using TensorFlow.js for real-time inference
│   ├── BISINDO.csv             # Dataset collected by the team
│   ├── notebook.ipynb          # Jupyter notebook for training and evaluation
│   ├── notebook.py             # Python script version of the notebook
│   ├── README.md               # Documentation for this directory
│   └── requirements.txt        # Dependencies for training
│
├── public-data-training/
│   ├── model/                  # Trained models using public BISINDO dataset
│   ├── web-inference/          # Web app using TensorFlow.js for this model
│   ├── BISINDO.zip             # Public BISINDO dataset
│   ├── notebook.ipynb          # Jupyter notebook for training with public dataset
│   ├── notebook.py             # Python script version of the notebook
│   ├── README.md               # Documentation for this directory
│   └── requirements.txt        # Dependencies for this setup
│
└── README.md                   # Main project documentation
```

---

## 🚀 Key Features

* Hand landmark extraction using **MediaPipe Hands**
* Alphabet recognition (A–Z) in **BISINDO**
* Classification using **Deep Neural Network (DNN)** and **1D Convolutional Neural Network (1D CNN)**
* **Real-time inference in the browser** using **TensorFlow\.js**
* Two training scenarios:

  * With team-collected dataset (`collected-data-training`)
  * With public BISINDO dataset (`public-data-training`)

---

## 🛠️ Technologies Used

* Python, TensorFlow, NumPy, Pandas
* MediaPipe (Hand Landmark Extraction)
* Jupyter Notebook
* TensorFlow\.js
* HTML, JavaScript (for web inference)

---

## 🚧 Roadmap / Future Work

* ✅ Hand landmark extraction using MediaPipe
* ✅ Train DNN and 1D CNN models on BISINDO alphabet dataset
* ✅ Real-time hand gesture inference using TensorFlow\.js
* ⬜ Expand dataset to include **numerical signs (0–9)**
* ⬜ Add **gesture-based word recognition** (basic common words)
* ⬜ Integrate other sign language datasets:

  * ⬜ **SIBI** (Sistem Isyarat Bahasa Indonesia)
  * ⬜ **ASL** (American Sign Language)
* ⬜ Evaluate models on more **diverse and larger datasets**

---

## 🙌 Final Words

ISYARA was built with the hope of making Sign Language more accessible and inclusive through the power of technology. Whether you're a developer, researcher, or simply someone passionate about sign language, we welcome you to explore, contribute, and improve this project.

Let’s work together to bridge communication gaps—one gesture at a time.

---
