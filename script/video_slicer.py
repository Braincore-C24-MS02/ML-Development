import cv2
import os

video_src = r'C:\Users\gedep\OneDrive\Documents\Github Repositories\ML-Development\dataset\video\driver_face_dataset.mp4'
target_folder = '../dataset/frames/'

# Test writing a txt file to target folder
# f = open(target_folder + 'test.txt', 'w')
# f.write('Hello World')
# f.close()

# Capture the video with cv2.VideoCapture
cap = cv2.VideoCapture(video_src)

# Count frames and pass them as file name
frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Write the frame to the target folder
    target_file = os.path.join(target_folder, f'{frame_count}.jpg')
    cv2.imwrite(target_file, frame)

    frame_count += 1

cap.release()