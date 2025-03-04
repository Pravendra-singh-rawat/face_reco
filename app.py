import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import os

def find_matches(video_path, target_image_path):
    # Load target image
    target_img = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)

    # ORB detector for feature matching
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(target_img, None)

    # Open video
    cap = cv2.VideoCapture(video_path)

    matches_log = []
    frame_count = 0

    # Create temp folder for saving captures
    output_folder = tempfile.mkdtemp()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect and compute features in the frame
        kp2, des2 = orb.detectAndCompute(gray_frame, None)

        if des2 is not None:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)

            # Good match threshold (tweak for accuracy)
            if len(matches) > 50:
                capture_path = os.path.join(output_folder, f"capture_{frame_count}.jpg")
                cv2.imwrite(capture_path, frame)

                matches_log.append({
                    'Frame Number': frame_count,
                    'Match Count': len(matches),
                    'Capture Path': capture_path
                })

    cap.release()
    return matches_log

st.title("🎥 Video Image Matcher")
st.write("Upload a video and a target image to find matching frames!")

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
        match_results = find_matches(video_path, image_path)

        if match_results:
            st.success("✅ Matches Found!")

            # Display matches
            for match in match_results:
                st.image(match['Capture Path'], caption=f"Frame: {match['Frame Number']} | Matches: {match['Match Count']}", use_column_width=True)

            # Log DataFrame
            df = pd.DataFrame(match_results)
            st.dataframe(df)

            # Download log
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Log", data=csv, file_name="match_log.csv", mime="text/csv")
        else:
            st.warning("❌ No significant matches found.")

    # Clean up
    os.remove(video_path)
    os.remove(image_path)
