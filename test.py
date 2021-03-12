import cv2 

def getFaceBox(net, frame, conf_threshold=0.5):
	frameOpencvDnn = frame.copy()
	frameHeight = frameOpencvDnn.shape[0]
	frameWidth = frameOpencvDnn.shape[1]
	blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

	net.setInput(blob)
	detections = net.forward()
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
	bboxes = []
	for i in range(detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > conf_threshold:
			x1 = int(detections[0, 0, i, 3] * frameWidth)
			y1 = int(detections[0, 0, i, 4] * frameHeight)
			x2 = int(detections[0, 0, i, 5] * frameWidth)
			y2 = int(detections[0, 0, i, 6] * frameHeight)
			bboxes.append([x1, y1, x2, y2])
			cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
	return frameOpencvDnn, bboxes

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
padding = 10

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

faceNet = cv2.dnn.readNet(faceModel, faceProto)

cap = cv2.VideoCapture(0)


while True:
	hasFrame, frame = cap.read()
	frameFace, bboxes = getFaceBox(faceNet, frame)
	for bbox in bboxes:
		face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
	cv2.imshow("Gender mood Demo", frameFace)
	k = cv2.waitKey(1)
	if k == 27:
		cap.release()
		cv2.destroyAllWindows()
		break
