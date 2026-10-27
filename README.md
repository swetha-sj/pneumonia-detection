# Pneumonia Detection from Chest X-ray Images Using CNN

## Project Overview
This project uses Deep Learning and a Convolutional Neural Network (CNN) to classify chest X-ray images into two categories: **NORMAL** and **PNEUMONIA**.

The trained model is integrated into a Streamlit web application that allows users to upload an X-ray image and view the model's prediction.

## Objectives
- Develop a CNN model for chest X-ray image classification.
- Preprocess images for deep learning.
- Train and evaluate the model using labeled X-ray images.
- Build an interactive web application using Streamlit.
- Display predictions and model scores.

## Technology Stack
- **Programming Language:** Python
- **Deep Learning:** TensorFlow and Keras
- **Model Architecture:** Convolutional Neural Network (CNN)
- **Image Processing:** Pillow and NumPy
- **Web Interface:** Streamlit
- **Visualization:** Matplotlib

## Model Architecture
The CNN consists of:
1. Input layer with image dimensions of 150 × 150 × 3
2. Rescaling layer
3. Three convolutional layers with 32, 64, and 128 filters
4. Max-pooling layers
5. Flatten layer
6. Dense layer with 128 neurons
7. Dropout layer with a rate of 0.5
8. Sigmoid output layer for binary classification

## Dataset
The dataset contains labeled chest X-ray images organized into two classes:
- NORMAL
- PNEUMONIA

The images are arranged into training, testing, and validation directories.

## Model Performance
- **Test Accuracy:** 73.24%
- **Test Loss:** 2.3342
- **Training:** 10 epochs
- **Final Training Accuracy:** Approximately 98.58%

The model currently produces false-positive predictions, so further evaluation and improvement are required.

## Project Structure
```text
pneumonia_detection/
├── app.py
├── train_model.py
├── evaluate_model.py
├── pneumonia_cnn.keras
├── training_results.png
├── confusion_matrix.png
├── README.md
└── requirements.txt
```

## How to Run the Project

### 1. Clone the repository
```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd pneumonia-detection-cnn
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
```

Windows:
```bash
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the application
```bash
streamlit run app.py
```

The application will open in your browser.

## Future Enhancements
- Improve model accuracy and generalization.
- Reduce false-positive predictions.
- Experiment with transfer learning.
- Add Grad-CAM visualizations to explain model predictions.
- Evaluate additional performance metrics.

## Disclaimer
This project is developed for educational purposes only. It is not a clinically validated diagnostic system and must not be used as a substitute for professional medical advice or diagnosis.