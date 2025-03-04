import streamlit as st
import cv2
import face_recognition
import numpy as np
import pandas as pd
import tempfile
import os

def find_faces(video_path, target_image_path):
    # Load target image and encode face
    target_image = face_recognition.load_image_file(target_image_path)
    target_encodings = face_recognition.face_encodings(target_image)

    if len(target_encodings) == 0:
        st.error("No face found in the target image. Please upload a valid image with a clear face.")
        return []

    target_encoding = target_encodings[0]

    # Open video
    cap = cv2.VideoCapture(video_path)

    matches_log = []
    frame_count = 0
    last_match_time = -10

    # Create temp folder for saving captures
    output_folder = tempfile.mkdtemp()

    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Convert BGR frame to RGB (required by face_recognition)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces and encode them
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Compare with target face
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces([target_encoding], face_encoding, tolerance=0.5)

            if True in matches:
                current_time = frame_count / fps

                if current_time - last_match_time >= 10:
                    capture_path = os.path.join(output_folder, f"capture_{frame_count}.jpg")
                    cv2.imwrite(capture_path, frame)

                    matches_log.append({
                        'Time (s)': round(current_time, 2),
                        'Capture Path': capture_path
                    })

                    last_match_time = current_time

    cap.release()
    return matches_log

st.title("🎥 Video Face Matcher")
st.write("Upload a video and a target image to find matching faces!")

video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
image_file = st.file_uploader("Upload Target Image", type=["jpg", "jpeg", "png"])

if video_file and image_file:
    with st.spinner("Processing... This may take a while."):
        # Save uploaded files temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as video_temp:
            video_temp.write(video_file.read())
            video_path = video_temp.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as image_temp:
            image_temp.write(image_file.read())
            image_path = image_temp.name

        # Process video and find matches
        match_results = find_faces(video_path, image_path)

        if match_results:
            st.success("✅ Matches Found!")

            # Display matches
            for match in match_results:
                st.image(match['Capture Path'], caption=f"Time: {match['Time (s)']}s", use_column_width=True)

            # Log DataFrame
            df = pd.DataFrame(match_results)
            st.dataframe(df)

            # Download log
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Log", data=csv, file_name="match_log.csv", mime="text/csv")
        else:
            st.warning("❌ No face matches found.")

    # Clean up
    os.remove(video_path)
    os.remove(image_path)
