# making streamlit app

import streamlit as st
import numpy as np
import librosa
import pickle
import os


MODEL_PATH = r"C:\Users\shubu\Desktop\Speaker Verification System\model.pkl"

#for test
TESTING_DATA = r"C:\Users\shubu\Desktop\Speaker Verification System\testing_data" 

#loading model
with open(MODEL_PATH, "rb") as f:
    clf = pickle.load(f)


# function to find MFCC statistics
SAMPLE_RATE = 16000
N_MFCC = 13

def MFCC_STATISTICS(filepath, sr=SAMPLE_RATE, n_mfcc=N_MFCC):
    
    wav, _ = librosa.load(filepath, sr=sr)   # load the file at 16 kHz
    wav, _ = librosa.effects.trim(wav)       # silence removed
    mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=n_mfcc)
    means = np.mean(mfcc, axis=1)                       
    stds  = np.std(mfcc, axis=1)                        
    return np.concatenate([means, stds])                


# streamlit app
def main():
    st.set_page_config(page_title="Speaker Verification", layout="centered")
    st.title("Speaker Verification")

    st.markdown(
        """
        This is a simple demo :
        1. Realtime audio (not supported in this project)  
        2. Predicting taget via audio files  
        """
    )

    # Sidebar: pick “mode”
    mode = st.sidebar.selectbox(
        "Choose prediction mode:", 
        ("File-based prediction", "Realtime")
    )

    if mode == "File-based prediction":
        show_file_based_ui(clf)
    else:
        show_live_audio_info()
        

# realtime prediction
def show_live_audio_info():

    st.header("Realtime Audio Prediction (Not Available)")
    st.markdown(
        """
        **Why not realtime capture:**  
        
        1. **Streamlit’s problem**: Streamlit not support microphone streaming.  
        
        2. Model trained on static dataset
        
        Try “File-based prediction” 
        """
    )


# prediction by selecting audios
def show_file_based_ui(clf):

    st.header("File-based Speaker Verification")

    # Subheading
    st.markdown(
        """
        Upload short audio clip.  
        We find MFCC stats for the audio clip and ask the model: “Is this the reference speaker?”  
        """
    )

    #  Upload your own audio file
    uploaded_file = st.file_uploader(
        label="Upload MP3 file",
        type=["wav", "mp3"]
    )

    # Choose from testing_data folder
    st.markdown("---")
    st.subheader("pick testing data: ")
    example_files = get_example_file_list(TESTING_DATA, limit=10)
    choice = st.selectbox(
        label="testing audio clips (from your training data)",
        options=["(none)"] + example_files
    )

    # prediction
    if uploaded_file is not None:
        # Save it to a temp file so librosa can load it
        with open("temp_uploaded_file", "wb") as f:
            f.write(uploaded_file.getbuffer())
        filepath = "temp_uploaded_file"

        st.audio(uploaded_file, format="audio/wav")

    elif choice != "(none)":
        filepath = os.path.join(TESTING_DATA, choice)
        if os.path.exists(filepath):
            st.audio(filepath, format="audio/wav")
        else:
            st.error("testing audio not found")
            return

    else:
        st.info("Upload audio or pick test audio above to see a prediction.")
        return

    # If we have a valid filepath, run prediction
    if st.button("Predict Speaker"):
        with st.spinner("Predicting..."):
            
            feat_vec = MFCC_STATISTICS(filepath)
            feat_vec = feat_vec.reshape(1, -1)  # shape → (1, 26)

            
            probs = clf.predict_proba(feat_vec)[0]
            pred_label = clf.predict(feat_vec)[0]       # 0 or 1
            confidence = probs[pred_label] * 100        # %

            label_str = "Target Speaker" if pred_label == 1 else "Non-Target Speaker"
            probability_str = f"{confidence:.1f}%"

            st.success(f" Prediction: **{label_str}** (confidence = {probability_str})")
            

#list testing audio files
def get_example_file_list(folder_path, limit=10):
    if not os.path.isdir(folder_path):
        return []

    all_files = [
        fn for fn in sorted(os.listdir(folder_path))
        if fn.lower().endswith((".wav", ".mp3"))
    ]
    return all_files[:limit]

#running app
if __name__ == "__main__":
    main()
