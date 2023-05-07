# import the necessary packages
import numpy as np
import time
import cv2

def run_ml_yolo(image, weight, cfg, classes_path):
	classes = open(classes_path).read().strip().split('\n')
	net = cv2.dnn.readNetFromDarknet(cfg, weight)
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)

	blob = cv2.dnn.blobFromImage(image, 1/255.0, (416,416), swapRB = True, crop = False)
	ln = net.getLayerNames()
	print(ln)
	print(net.getUnconnectedOutLayers())
	for i in net.getUnconnectedOutLayers():
		print(i)
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
	print(ln)
	net.setInput(blob)
	start = time.time()
	outputs = net.forward(ln)
	end = time.time()
	print("[INFO] yolo took {:.5} seconds".format(end - start))
	boxes = []
	confidences = []
	classIDs = []
	imgh,imgw = image.shape[:2]
	print(imgh,imgw)
	print("yolo: ", len(outputs))
	for out in outputs:
		print(out.shape)
		for detection in out:
			score = detection[5:]
			classID = np.argmax(score)
			confidence = score[classID]
			if confidence > 0.5:
				print(detection[:4])
				box = detection[:4] * 416
				(centerX, centerY, width, height) = box.astype("int")
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				box = [int(x * imgw/416), int(y*imgh/416), int(width*imgw/416), int(height*imgh/416)]
				boxes.append(box)
				confidences.append(float(confidence))
				classIDs.append(classID)
	print(len(boxes), boxes)
	print(len(confidences), confidences)
	print(len(classIDs), classIDs)
	if (len(boxes) == 0):
		return image
	indices = []
	#for box in enumerate(boxes):
	indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.6)
	print(indices)
	if len(indices) > 0:
		for i in indices.flatten():
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			#color = [int(c) for c in colors[classIDs[i]]]
			cv2.rectangle(image, (x, y), (x + w, y + h), (255,255,255), 2)
			text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
			cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
	return 
def run_ml_yoloonnx(image, onnx):
	IMG_SIZE = 640
	#classes = open(classes_path).read().strip().split('\n')
	print(cv2.__version__)
	net = cv2.dnn.readNet(onnx)
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
	print('net loaded')
	#print(image.shape)
	blob = cv2.dnn.blobFromImage(image, 1/255.0, size = (IMG_SIZE,IMG_SIZE),swapRB = True, crop = False)
	ln = net.getLayerNames()
	#print(net.getUnconnectedOutLayers())
	for i in net.getUnconnectedOutLayers():
		print(i)
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
	#print(ln)
	net.setInput(blob)
	#print(blob.shape)
	start = time.time()
	outputs = net.forward()
	end = time.time()
	print("[INFO] yolo took {:.5} seconds".format(end - start))
	boxes = []
	confidences = []
	classIDs = []
	imgh,imgw = image.shape[:2]
	print(imgh,imgw)
	print("yolo: ", len(outputs))
	print(outputs[0].shape)
	for out in outputs:
		#print(out.shape)
		for detection in out:
			confidence = detection[4]
			if confidence > 0.5:
				#print(detection[:4])
				scores = detection[5:]
				classID = np.argmax(scores)
				if scores[classID] > 0.25:
					print(detection)
					confidences.append(float(confidence))
					classIDs.append(classID)
					x,y,w,h = detection[:4]
					left = int((x-.5*w) * imgw/IMG_SIZE)
					top = int((y-.5*h) * imgh/IMG_SIZE)
					width = int(w * imgw/IMG_SIZE)
					height = int(h * imgh/IMG_SIZE)
					box = np.array([left, top, width, height])
					boxes.append(box)
				
				
	print(len(boxes), boxes)
	print(len(confidences), confidences)
	print(len(classIDs), classIDs)
	if (len(boxes) == 0):
		return image
	indices = []
	#for box in enumerate(boxes):
	indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.3)
	print(indices)
	if len(indices) > 0:
		for i in indices.flatten():
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			#color = [int(c) for c in colors[classIDs[i]]]
			cv2.rectangle(image, (x, y), (x + w, y + h), (255,255,255), 2)
			#text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
			#cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
	return image
#
#def run_ml_caffe(image, model, prototxt, labels):
#	# load the input image from disk
#	#image = cv2.imread(args["image"])
#	# load the class labels from disk
#	rows = open(labels).read().strip().split("\n")
#	classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]
#	print(len(classes))
## our CNN requires fixed spatial dimensions for our input image(s)
## so we need to ensure it is resized to 224x224 pixels while
## performing mean subtraction (104, 117, 123) to normalize the input;
## after executing this command our "blob" now has the shape:
## (1, 3, 224, 224)
#	blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))
#	print(blob)
## load our serialized model from disk
#	print("[INFO] loading model...")
#	net = cv2.dnn.readNetFromCaffe(prototxt, model)
#
## set the blob as input to the network and perform a forward-pass to
## obtain our output classification
#	net.setInput(blob)
#	start = time.time()
#	preds = net.forward()
#	end = time.time()
#	print("[INFO] classification took {:.5} seconds".format(end - start))
#
## sort the indexes of the probabilities in descending order (higher
## probabilitiy first) and grab the top-5 predictions
#	idxs = np.argsort(preds[0])[::-1][:5]
#
## loop over the top-5 predictions and display them
#	for (i, idx) in enumerate(idxs):
#		# draw the top prediction on the input image
#		if i == 0:
#			text = "Label: {}, {:.2f}%".format(classes[idx],
#				preds[0][idx] * 100)
#			cv2.putText(image, text, (5, 25),  cv2.FONT_HERSHEY_SIMPLEX,
#				0.7, (0, 0, 255), 2)
#		# display the predicted label + associated probability to the
#		# console	
#		print("[INFO] {}. label: {}, probability: {:.5}".format(i + 1,
#			classes[idx], preds[0][idx]))
#	# display the output image
#	return image
#
# https://pyimagesearch.com/2017/08/21/deep-learning-with-opencv/
# python deep_learning_with_opencv.py --image /home/lixin/Classes/Fall22Lab/kivyapp/ml/isopod.jpg --prototxt bvlc_googlenet.prototxt --model bvlc_googlenet.caffemodel --labels synset_words.txt