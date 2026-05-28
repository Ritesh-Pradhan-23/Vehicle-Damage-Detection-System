# 🚗 Vehicle Damage Detection System

[cite_start]This project builds a binary image classification model using CNN (VGG16 transfer learning) to automatically detect whether a vehicle is damaged or undamaged[cite: 32]. [cite_start]Manual inspection for insurance claims is time-consuming, so this AI-powered assessment provides a scalable and rapid solution[cite: 31, 32].

## 📊 Dataset & Model
* [cite_start]**Dataset**: Custom car image dataset containing 18,992 total images (15,000 for training, 3,992 for testing)[cite: 35, 36, 39, 41].
* [cite_start]**Classes**: Undamaged (Class 0) and Damaged (Class 1)[cite: 70, 71, 72, 73].
* [cite_start]**Architecture**: Pre-trained VGG16 base (with frozen weights) connected to a custom classification head (Flatten → Dense 256 → Dropout 0.5 → Dense 2 with Softmax)[cite: 106, 110, 111, 130, 132, 134, 136].
* [cite_start]**Performance**: Achieved ~97% training accuracy using the Adam optimizer and Sparse Categorical Crossentropy loss over 5 epochs[cite: 75, 76, 77, 78, 79, 80, 167, 168].

## 🛠️ Tech Stack
* [cite_start]**Deep Learning**: TensorFlow & Keras[cite: 48].
* [cite_start]**Computer Vision**: OpenCV (`cv2`) for image reading, resizing to 224x224, and BGR preprocessing[cite: 54, 55, 93, 94].
* [cite_start]**Frontend UI**: Streamlit for real-time web deployment[cite: 60, 61].
* [cite_start]**Data Processing**: NumPy & Pandas[cite: 57, 58].

## 🚀 Streamlit Web App Features
[cite_start]The project includes a live web application where users can upload an image (JPG, JPEG, PNG up to 200MB)[cite: 139, 153, 154]. 
* [cite_start]**Cached Model Loading**: Ensures the VGG16 model only loads once, allowing for fast repeated inference[cite: 155, 156].
* [cite_start]**Confidence Slider**: An adjustable threshold in the sidebar to set the minimum confidence for a valid prediction[cite: 159, 160].
* [cite_start]**Visual Feedback**: Displays green success alerts for "No Damage Detected" and red warnings for "Damaged Vehicle Detected"[cite: 143, 149, 161, 162].
* [cite_start]**Channel Consistency**: The app automatically reverses RGB inputs to BGR to match the `cv2.imread()` training pipeline, ensuring accurate predictions[cite: 163, 164].

## ⚙️ How to Run Locally

1. **Clone the repository**:
   git clone https://github.com/YourUsername/Vehicle-Damage-Detection.git
   cd Vehicle-Damage-Detection

2. **Install the required dependencies**:
   pip install -r requirements.txt

3. **Run the Streamlit application**:
   streamlit run Car_damage_app_01.py
