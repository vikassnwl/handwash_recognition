import streamlit as st
import os
import requests

# Directory to save uploaded videos
save_directory = "uploaded_videos"
os.makedirs(save_directory, exist_ok=True)  # Create the directory if it doesn't exist

# Streamlit UI
# st.title("Upload a Video and Save to Directory")
st.title("Upload a Video")

# File uploader widget for video files
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

# Save the uploaded video
if uploaded_file is not None:
    # Display the uploaded video in Streamlit
    st.video(uploaded_file)
    
    # Save the video file in the specified directory
    # file_path = os.path.join(save_directory, uploaded_file.name)
    file_path = os.path.join(save_directory, "test_video.mp4")
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f"Video saved successfully in {file_path}")

    # API call
    backend_url = "http://127.0.0.1:5000/predict"
    response = requests.get(backend_url)
    st.write(response.text)