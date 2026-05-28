# 🚗 Vehicle Damage Detection System

![Vehicle Damage App Screenshot](Images/Screenshot%202026-05-29%20013326.png)

This project builds a binary image classification model using CNN (VGG16 transfer learning) to automatically detect whether a vehicle is damaged or undamaged. Manual inspection for insurance claims is time-consuming, so this AI-powered assessment provides a scalable and rapid solution.

## 📊 Dataset & Model
* **Dataset**: Custom car image dataset containing 18,992 total images (15,000 for training, 3,992 for testing).
* **Classes**: Undamaged (Class 0) and Damaged (Class 1).
* **Architecture**: Pre-trained VGG16 base (with frozen weights) connected to a custom classification head (Flatten → Dense 256 → Dropout 0.5 → Dense 2 with Softmax).
* **Performance**: Achieved ~97% training accuracy using the Adam optimizer and Sparse Categorical Crossentropy loss over 5 epochs.

## 🛠️ Tech Stack
* **Deep Learning**: TensorFlow & Keras.
* **Computer Vision**: OpenCV (`cv2`) for image reading, resizing to 224x224, and BGR preprocessing.
* **Frontend UI**: Streamlit for real-time web deployment.
* **Data Processing**: NumPy & Pandas.

## 🚀 Streamlit Web App Features
The project includes a live web application where users can upload an image (JPG, JPEG, PNG up to 200MB). 
* **Cached Model Loading**: Ensures the VGG16 model only loads once, allowing for fast repeated inference.
* **Confidence Slider**: An adjustable threshold in the sidebar to set the minimum confidence for a valid prediction.
* **Visual Feedback**: Displays green success alerts for "No Damage Detected" and red warnings for "Damaged Vehicle Detected".
* **Channel Consistency**: The app automatically reverses RGB inputs to BGR to match the `cv2.imread()` training pipeline, ensuring accurate predictions.

## ⚙️ How to Run Locally

1. **Clone the repository**:
```bash
   git clone [https://github.com/Ritesh-Pradhan-23/Vehicle-Damage-Detection.git](https://github.com/Ritesh-Pradhan-23/Vehicle-Damage-Detection.git)
   cd Vehicle-Damage-Detection
