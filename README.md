# 😊 Emotion Detection Web App

This project is a web-based **Emotion Detection Application** that leverages **Natural Language Processing (NLP)** and **Facial Emotion Recognition** to analyze and predict emotions from text or real-time video input.

---

## 🌟 Features

* **Text Emotion Analysis**:
  - Enter text or use speech input to analyze the emotion.
  - Outputs a predicted emotion along with a confidence score.
  - Displays a bar chart of prediction probabilities.

* **Real-Time Webcam Emotion Detection** *(currently disabled)*:
  - Detects emotions using a webcam feed.
  - Employs **OpenCV** and **FER** for real-time emotion detection.

* **Feedback Form**:
  - Users can submit feedback or suggestions directly through the app.

---

## 🛠️ Technology Stack

* **Programming Language**: Python
* **Framework**: Streamlit
* **NLP Libraries**: NLTK, HuggingFace Transformers
* **Emotion Recognition Libraries**: OpenCV, FER, TensorFlow
* **Visualization**: Altair
* **Other Tools**: Joblib, SpeechRecognition, Pandas, NumPy

---

## 📂 Directory Structure

```plaintext
📁 EmotionDetectionApp
├── 📁 model
│   └── emotion_classifier_pipe_lr.pkl  # Pre-trained emotion detection model
├── 📄 app.py  # Main application code
└── 📄 requirements.txt  # List of dependencies

- `model/emotion_classifier_pipe_lr.pkl`: Pre-trained emotion classification model.
- `app.py`: Main application code.
- `requirements.txt`: Python dependencies for the project.
- `README.md`: Project documentation.

---

## 🚀 How to Run

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Yashrajgithub/Emotion-Recognization-and-Analysis.git

2. **Run the Application**:
   ```bash
   streamlit run app.py

3. **Access the App:**
   ```bash
   usually http://localhost:8501

---

## 📖 ***Usage***

### **_Home:_**
* Choose between **"Text Recognition"** or **"Webcam Emotion Detection."**
* Enter text, or use the **speech-to-text** functionality to predict emotions.

---

### **_About:_**
* Learn about the project's purpose, technology stack, and functionality.

---

### **_Feedback:_**
* Submit feedback or suggestions via the provided form.
