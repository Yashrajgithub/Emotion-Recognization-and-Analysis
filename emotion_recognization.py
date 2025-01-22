import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
# import cv2  # Comment out OpenCV import
# from fer import FER  # Comment out FER import
import speech_recognition as sr
import requests
import os

# Define the model path
model_path = os.path.join(os.getcwd(), "model", "emotion_classifier_pipe_lr.pkl")

# Check if the model file exists
if os.path.exists(model_path):
    with open(model_path, "rb") as file:
        pipe_lr = joblib.load(file)
else:
    st.error("Model file not found!")

# Dictionary for emotions and corresponding emojis
emotions_emoji_dict = {
    "anger": "😠",
    "disgust": "🤮",
    "fear": "😨😱",
    "happy": "🤗",
    "joy": "😂",
    "neutral": "😐",
    "sad": "😔",
    "sadness": "😔",
    "shame": "😳",
    "surprise": "😮"
}

# Apply custom font styling (Times New Roman)
st.markdown(
    """
    <style>
    body {
        font-family: "Times New Roman", serif;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to predict emotion
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

# Function to get prediction probabilities
def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

# Speech recognition function
def speech_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = r.listen(source)

    try:
        text = r.recognize_google(audio)
        st.write("Transcription:", text)
        return text
    except sr.UnknownValueError:
        st.write("Google Speech Recognition could not understand audio")
        return ""
    except sr.RequestError as e:
        st.write("Could not request results from Google Speech Recognition service; {0}".format(e))
        return ""
    except Exception as e:
        st.write("No audio or other error:", str(e))
        return ""

# Function to determine if the detected object is a mobile device
def is_mobile_device(face):
    height, width = face.shape[:2]
    aspect_ratio = width / height
    return aspect_ratio > 1.5

# Commented out real-time emotion detection using webcam
# def detect_emotions():
#     # Initialize FER
#     detector = FER()
#     cap = cv2.VideoCapture(0)
#     # Placeholder for the video feed
#     video_placeholder = st.empty()
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         emotion, score = detector.top_emotion(frame)
#         if emotion is not None and score is not None:
#             if is_mobile_device(frame):
#                 cv2.putText(frame, "Mobile Device Detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             else:
#                 cv2.putText(frame, f"{emotion} ({score:.2f})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#         else:
#             cv2.putText(frame, "No emotion detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         video_placeholder.image(frame, channels="RGB", use_container_width=True, caption="Real-time Emotion Detection")
#     cap.release()
#     video_placeholder.empty()

# Feedback form submission function
def submit_feedback(name, email, feedback):
    url = "https://api.web3forms.com/submit"
    payload = {
        "access_key": "4ffcbd0a-8334-41a7-af0a-d8552c02dd27",
        "name": name,
        "email": email,
        "message": feedback
    }
    response = requests.post(url, data=payload)
    return response

# Sidebar menu
def sidebar():
    st.sidebar.title(" 😀 Menu")
    page = st.sidebar.radio("Select an Option", ("Home", "About", "Feedback/Suggestions"))
    return page

# Home page
def home():
    st.title("😊 Emotion Detection 😊")
    st.markdown("#### **Analyze the emotions expressed in your text or via webcam!**")

    # Sidebar for choosing the analysis type
    analysis_option = st.sidebar.radio("Choose an Analysis Option", 
                                       ("Text Recognition", "Webcam Emotion Detection"))

    if analysis_option == "Text Recognition":
        st.subheader("**Enter Text to Analyze Emotion**")
        with st.form(key='text_form', clear_on_submit=True):
            raw_text = st.text_area("**Enter Your Text Here**", height=150)
            submit_text = st.form_submit_button(label='Analyze Emotion')
            speak_checkbox = st.checkbox("Use Speech Input")
            if speak_checkbox:
                spoken_text = speech_to_text()
                if spoken_text:
                    raw_text = spoken_text

        if submit_text:
            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### **Original Text:**")
                st.success(raw_text)
                st.markdown("### **Predicted Emotion:**")
                emoji_icon = emotions_emoji_dict[prediction]
                st.subheader(f"{prediction.capitalize()} {emoji_icon}")
                st.markdown(f"**Confidence:** {np.max(probability):.2f}")

            with col2:
                st.markdown("### **Prediction Probability:**")
                proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["Emotions", "Probability"]

                fig = alt.Chart(proba_df_clean).mark_bar().encode(
                    x='Emotions',
                    y='Probability',
                    color=alt.Color('Probability', scale=alt.Scale(scheme='blues')),  # Apply color gradient
                    tooltip=['Emotions', 'Probability']
                ).properties(width=400, height=300)

                st.altair_chart(fig, use_container_width=True)

            st.markdown("---")
            st.markdown("### **Keep Exploring Your Emotions!**")
            st.markdown("✨ **Embrace your feelings!** ✨")

    elif analysis_option == "Webcam Emotion Detection":
        st.subheader("**Real-Time Webcam Emotion Detection**")
        st.warning(
            "The webcam feature is currently unavailable due to system compatibility issues or errors. "
            "You can uncomment the webcam section in the code and run it on your local machine."
        )
        # Commented out the call to detect_emotions
        # detect_emotions()

# About page
def about():
    st.title(" ℹ️ About This Project")
    st.markdown("""
        This project combines **Natural Language Processing (NLP)** and **Facial Emotion Recognition** to detect and analyze emotions from text and video input.

        **Text Emotion Analysis**: 
        - Utilizes a pre-trained model to predict emotions from text, identifying feelings like happiness, sadness, and anger.

        **Real-Time Facial Emotion Recognition**: 
        - Uses **FER** and **OpenCV** to detect emotions from a user's face via webcam in real-time.

        **Technology Stack**:
        - **NLP**: NLTK, TextBlob, HuggingFace Transformers
        - **Emotion Recognition**: FER, OpenCV, TensorFlow
        - **Web Framework**: Streamlit

        This project offers a unique approach to analyzing emotions, useful in fields like customer service and mental health.

        **Developed by**: Yashraj Kalshetti 🧑‍🎓
    """)

# Feedback page
def feedback():
    st.title("Feedback / Suggestions 🤗 ")
    st.markdown(" 🤝 We value your feedback! Please provide your suggestions or feedback below:")

    with st.form(key='feedback_form', clear_on_submit=True):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        feedback_text = st.text_area("Your Feedback/Suggestions", height=150)
        submit_feedback_btn = st.form_submit_button(label='Submit Feedback')
        
        if submit_feedback_btn:
            if name and email and feedback_text:
                response = submit_feedback(name, email, feedback_text)
                if response.status_code == 200:
                    st.success("Your feedback has been submitted successfully! Thank you.")
                else:
                    st.error("Failed to submit feedback. Please try again later.")
            else:
                st.warning("Please fill out all fields.")

# Main function to run the app
def main():
    page = sidebar()

    if page == "Home":
        home()
    elif page == "About":
        about()
    elif page == "Feedback/Suggestions":
        feedback()

# Run the app
if __name__ == '__main__':
    main()
