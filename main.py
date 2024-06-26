from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2

# Define eye aspect ratio formula
def calculate_EAR(eye):
    ver1 = distance.euclidean(eye[1], eye[5])
    ver2 = distance.euclidean(eye[2], eye[4])
    hor = distance.euclidean(eye[0], eye[3])
    ear = (ver1 + ver2) / (2.0 * hor)
    return ear

# Define mouth aspect ratio formula
def mouth_aspect_ratio(mouth):
    ver1 = distance.euclidean(mouth[1], mouth[7])
    ver2 = distance.euclidean(mouth[2], mouth[6])
    ver3 = distance.euclidean(mouth[3], mouth[5])
    hor = distance.euclidean(mouth[0], mouth[4])
    mar = (ver1 + ver2 + ver3) / (2.0 * hor)
    return mar

# Define facial landmark indices
thresh_ear = 0.25  # Eye aspect ratio threshold
thresh_mar = 1.2  # Mouth aspect ratio threshold
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")# Dat file is the crux of the code

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

cap=cv2.VideoCapture(0)
flag=0
while True:
	ret, frame=cap.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detect faces
	subjects = detect(gray, 0)
	if not subjects:
		print("No face detected in the frame.")
		continue

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
		mar = mouth_aspect_ratio(mouth)

		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		mouthHull = cv2.convexHull(mouth)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

		cv2.putText(frame, f'Left EAR: {leftEAR:.2f}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
		cv2.putText(frame, f'Right EAR: {rightEAR:.2f}', (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
		cv2.putText(frame, f'MAR: {mar:.2f}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 0), 2, cv2.LINE_AA)

		drowsy = False
		if ear < thresh_ear or mar > thresh_mar:
			flag += 1
			drowsy = True
		else:
			flag = 0
		
		if drowsy:
			if ear < thresh_ear and mar > thresh_mar:
				text = "Drowsy (eye and mouth)"
			elif ear <= thresh_ear:
				text = "Drowsy (eye)"
			else:
				text = "Drowsy (mouth)"
			cv2.putText(frame, text, (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
		else:
			cv2.putText(frame, "OK", (375, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
			
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
cv2.destroyAllWindows()
cap.release() 