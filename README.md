# ML-Development

Reference:
1. Dataset from RoboFlow https://universe.roboflow.com/driver-wlf6m/driving-monitoring-system/dataset/2
2. Drowsiness Detection https://github.com/akshaybahadur21/Drowsiness_Detection/blob/master/Drowsiness_Detection.py

<br>Before starting EAR and MAR, you should download shape_predictor.dat through https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat

<br>Driver Monitoring System is consisted of two models
1. Object detection, trained by YOLOv8, detects bottle, phone, smoke, vape. You can access it in `best.pt` file
2. Face detection, trained using cvzone and shape_predictor.dat, detects yawn through EAR (eye aspect ratio) and MAR (mouth aspect ratio). You can access it in `main.py`

<br>The Driver Monitoring System can be run in `dms.py` file
