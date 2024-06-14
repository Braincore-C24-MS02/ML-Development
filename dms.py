import cv2
import dlib
import imutils
import numpy as np
from imutils import face_utils
from scipy.spatial import distance
from ultralytics import YOLO
import time
import io
import requests
import json
import os

# Define eye aspect ratio formula
def calculate_EAR(eye):
    ver1 = distance.euclidean(eye[1], eye[5])
    ver2 = distance.euclidean(eye[2], eye[4])
    hor = distance.euclidean(eye[0], eye[3])
    ear = (ver1 + ver2) / (2.0 * hor)
    return ear

# Define mouth aspect ratio formula
def calculate_MAR(mouth):
    ver1 = distance.euclidean(mouth[2], mouth[10])
    ver2 = distance.euclidean(mouth[3], mouth[9])
    ver3 = distance.euclidean(mouth[4], mouth[8])
    hor = distance.euclidean(mouth[0], mouth[6])
    mar = (ver1 + ver2 + ver3) / (2.0 * hor)
    return mar

def convert_frames_to_video(frames, width, height, fps=10):
    # Ensure there are frames to process
    if not frames:
        raise ValueError("No frames provided to convert to video.")

    size = (frames[0].shape[1], frames[0].shape[0])  # Get the size of the frames (width, height)
    
    fourcc = cv2.VideoWriter_fourcc('D','I','V','X')
    temp_video_path = 'temp_video.avi'
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, size)

    # Check if VideoWriter object was created successfully
    if not out.isOpened():
        raise RuntimeError("Error opening video writer")

    for frame in frames:
        out.write(frame)
    
    out.release()

    # Check if the file was written successfully
    if not os.path.exists(temp_video_path):
        raise RuntimeError("Video file was not created")

    with open(temp_video_path, 'rb') as video_file:
        video_bytes = video_file.read()
    
    return video_bytes

def send_video_to_api(video_bytes, url):
    files = {'video': ('video.mp4', io.BytesIO(video_bytes), 'video/mp4')}
    response = requests.post(url, files=files)
    return response.json()

def send_data_to_collection_api(data, url):
    params = {
        "collection": "alerts"
    }
    try:
        response = requests.post(url, json=data, params=params)
        response.raise_for_status()  # Raise an HTTPError for bad responses
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

    try:
        return response.json()  # Attempt to parse JSON response
    except ValueError as e:
        print(f"Failed to parse JSON response: {e}")
        print("Response content:", response.content)
        return None

# Define facial landmark indices
thresh_ear = 0.25  # Eye aspect ratio threshold
thresh_mar = 1.25  # Mouth aspect ratio threshold
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

# Load the YOLO model
model_path = 'best.pt'
model = YOLO(model_path)

# # Open the video capture
# cap = cv2.VideoCapture(0)

url = r'http://34.34.218.86:5000'
cap = cv2.VideoCapture(url)

# Create requests to open the stream from URL

# Initialize variables for accumulating results
ear_list = []
mar_list = []
object_detected_list = []
frame_list = []
drowsy_frames = []
missing_face_frames = []

width = int(cap.get(3))
height = int(cap.get(4))

start_time = time.time()

while True:
    try:
        is_frame_drowsy = False
        
        ret, frame = cap.read()
        print("frame is type: ", type(frame))
        if not ret:
            break
        frame = imutils.resize(frame, width=640)
        print("frame is type: ", type(frame))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        frame_list.append(frame)

        # Detect faces
        subjects = detect(gray, 0)
        if not subjects:
            print("No face detected in the frame.")
            missing_face_frames.append(frame)

        for subject in subjects:
            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)  # Convert to NumPy array

            # Extract eye and mouth landmarks
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            mouth = shape[mStart:mEnd]

            # Calculate EAR and MAR
            leftEAR = calculate_EAR(leftEye)
            rightEAR = calculate_EAR(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            mar = calculate_MAR(mouth)

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            mouthHull = cv2.convexHull(mouth)
            # cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            # cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            # cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

            # cv2.putText(frame, f'Left EAR: {leftEAR:.2f}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
            # cv2.putText(frame, f'Right EAR: {rightEAR:.2f}', (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
            # cv2.putText(frame, f'MAR: {mar:.2f}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 0), 2, cv2.LINE_AA)

            if ear < thresh_ear:
                ear_list.append(ear)
                is_frame_drowsy = True
                print(f"EAR below threshold: {ear}")
            if mar > thresh_mar:
                mar_list.append(mar)
                is_frame_drowsy = True
                print(f"MAR above threshold: {mar}")

        print("Current frame count: ", len(frame_list))

        # Object detection
        results = model(frame)
        alert_message = ""

        detected_object = False
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                if label in ["bottle", "phone", "vape", "smoke"]:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    if confidence > 0.5:
                        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # cv2.putText(frame, f'{label}: {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        detected_object = True
                        alert_message += f'{label} detected! '

        object_detected_list.append(detected_object)
        if detected_object:
            is_frame_drowsy = True

        # print("Current frame count: ", len(frame_list))

        # if alert_message:
        #     cv2.putText(frame, alert_message, (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if is_frame_drowsy:
            drowsy_frames.append(frame)

        current_time = time.time()
        elapsed_time = current_time - start_time
        
        # print("Current frame count: ", len(frame_list))
        
        # print("Total frames has type: ", type(frame_list))
        # print("ear_list has type: ", type(ear_list))
        # print("mar_list has type: ", type(mar_list))
        # print("object_detected_list has type: ", type(object_detected_list))
        # print("drowsy_frames has type: ", type(drowsy_frames))
        # print("frame has type: ", type(frame))

        if len(frame_list) >= 70:  # Check results every 70 frames (7 seconds)
            total_frames = len(frame_list)  # Total frames processed in 70 frames (7 seconds)
            drowsy_ear_frames = len(ear_list)
            drowsy_mar_frames = len(mar_list)
            detected_objects_frames = sum(object_detected_list)
            
            # Calculate the number of OK frames
            drowsy_frames_count = len(drowsy_frames)
            ok_frames = total_frames - drowsy_frames_count
            
            undetected_face = False
            drowsy = False
            if total_frames > 0 and drowsy_frames_count / total_frames >= 0.6:  # More than 60% of frames show drowsiness/distraction
                drowsy = True
            
            if len(missing_face_frames) / total_frames >= 0.6:
                undetected_face = True
                
            send_api_data = False

            message_text = ""

            if drowsy:
                send_api_data = True
                if drowsy_ear_frames > drowsy_mar_frames and drowsy_ear_frames > detected_objects_frames:
                    # text = "Drowsy (eye)"
                    # # print(f"Drowsy (eye) detected for {drowsy_ear_frames} EAR frames out of {total_frames} frames in the last 10 seconds")
                    # print(f"DROWSY (eye) detected. Please rest!")
                    message_text = "Drowsy (eye) detected. Please rest!"
                elif drowsy_mar_frames > drowsy_ear_frames and drowsy_mar_frames > detected_objects_frames:
                    # text = "Drowsy (mouth)"
                    # # print(f"Drowsy (mouth) detected for {drowsy_mar_frames} MAR frames out of {total_frames} frames in the last 10 seconds")
                    # print(f"DROWSY (mouth) detected. Please rest!")
                    message_text = "Drowsy (mouth) detected. Please rest!"
                elif detected_objects_frames > drowsy_ear_frames and detected_objects_frames > drowsy_mar_frames:
                    # text = "Distracted (object detected)"
                    # # print(f"Distracted (object detected) for {detected_objects_frames} frames out of {total_frames} frames in the last 10 seconds")
                    # print(f"DISTRACTED (object detected) detected. Please rest!")
                    message_text = "Distracted (object detected) detected. Please focus!"
                else:
                    # text = "Drowsy (eye and mouth)"
                    # # print(f"Drowsy (eye and mouth) detected for {drowsy_ear_frames} EAR frames and {drowsy_mar_frames} MAR frames out of {total_frames} frames in the last 10 seconds")
                    # print(f"DROWSY (mouth and eye) detected. Please rest!")
                    message_text = "Drowsy (mouth and eye) detected. Please rest!"
                # cv2.putText(frame, text, (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif undetected_face:
                send_api_data = True
                message_text = "Face not detected. Please adjust your position!"
            else:
                # print(f"Driver is OK for the last {ok_frames} frames out of {total_frames} frames in the last 10 seconds")
                print(f"Driver is OK")
                # cv2.putText(frame, "OK", (550, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if send_api_data:
                # Send data to API
                api_url_firestore = "http://34.101.117.35:8000/send/send-to-firestore"
                api_url_storage = "http://34.101.117.35:8000/send/send-data"
                video_bytes = convert_frames_to_video(frame_list, width, height)
                saved_url = send_video_to_api(video_bytes, api_url_storage)
                print(saved_url)
                print(saved_url['url'])
                data = {
                    "driver_id": 1,
                    "video_url": saved_url['url'],
                    "message": message_text
                }
                response = send_data_to_collection_api(data, api_url_firestore)
                print("the error is after this one")
                print(response)

            # Reset lists and timer
            ear_list.clear()
            mar_list.clear()
            object_detected_list.clear()
            frame_list = []
            drowsy_frames = []
            start_time = time.time()
        
        # cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    except Exception as e:
        exit(e)

# cv2.destroyAllWindows()
cap.release()
