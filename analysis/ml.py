import numpy as np
import time
import cv2
import math
import sys
import os
import tflite_runtime.interpreter as tflite
import matplotlib.pyplot as plt
def init_model(path):
    interpreter = tflite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter
def run_ml_tflite(image, interpreter,input_index, conf=0.5):
    MODEL_INPUT_SIZE = 640
    image = image.astype(np.float32)
    image = image/255.
    input_data = np.expand_dims(image,axis=0)
    print(image.shape)
    #plt.imshow(input_data[0])
    #plt.show()
    interpreter.set_tensor(input_index,input_data)
    interpreter.invoke()
    
    output_details = interpreter.get_output_details()

    out = np.squeeze(interpreter.get_tensor(output_details[0]['index']))
    #classes = np.squeeze(interpreter.get_tensor(output_details[1]['index']))
    #scores = np.squeeze(interpreter.get_tensor(output_details[2]['index']))
    #print(out.shape)
    boxes = []
    confidences = []
    classIDs = []
    radii = []
    imgh,imgw = image.shape[:2]
    for detection in out:
        confidence = detection[4]
        if confidence > conf:
    			#print(detection[:4])
            scores = detection[5:]
            classID = np.argmax(scores)
            if scores[classID] > conf:
                #print(detection)
                confidences.append(float(confidence))
                classIDs.append(classID)
                x,y,w,h = detection[:4]
                left = int((x-.5*w) * imgw)
                top = int((y-.5*h) * imgh)
                width = math.ceil(w * imgw)
                height = math.ceil(h * imgh)
                box = np.array([left, top, width, height])
                boxes.append(box)

    indices = []
    #for box in enumerate(boxes):
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf, 0.2)
    #print(indices)

    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            #color = [int(c) for c in colors[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0,255,0), 1)
            radii.append(np.min([w,h])/2)     # results
            #text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
            #cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    #print("tests: ", image.shape)
    #print("tests: ", np.average(radii))
    #image = image[:,80:-80,:]
    return image, np.array(radii)
def GeneratOutput(frame, interpreter, input_index, input_shape, conf = 0.5):
    frame_size = frame.shape[:2]
    max_side = np.argmax(frame_size)
    #print(max_side,frame_size[max_side])
    resize_ratio = input_shape[max_side+1]/frame_size[max_side]
    new_shape = (int(frame_size[1]*resize_ratio), int(frame_size[0]*resize_ratio))
    frame = cv2.resize(frame, new_shape)
    frame = cv2.copyMakeBorder(frame,0,0,0,160,cv2.BORDER_REPLICATE)

    #print(frame.shape)
    results, radii = run_ml_tflite(frame,interpreter, input_index,conf)

    # used for results (found wrong after the results were recorded, corrected in excel sheet)
    #avg_area = np.mean(np.pi * np.power(radii,2))/resize_ratio 

    # correct pixel area at original 4032x3024 image
    avg_area = np.mean(np.pi * np.power(radii,2))/np.power(resize_ratio,2) 

    print("\taverage area (see code comments): ", avg_area)
    print("\tcount: ", len(radii))
    #plt.imshow(results)
    #plt.show()
    #plt.close()
    return len(radii), avg_area
if __name__ == "__main__":
    img_dirs=[]
    model_path = '/home/lixin/Classes/Fall22Lab/kivyapp/ml/yolov5m-best-fp16.tflite'
    #img_folder = "/home/lixin/Classes/Fall22Lab/drive-download-20220904T003514Z-001/"
    #folder = "auburn"
    #img_folder = "/home/lixin/Classes/Fall22Lab/Fairhope_052322/"
    #folder = "fairhope"
    #img_folder = "/home/lixin/Classes/Fall22Lab/Data2023-20230714T162453Z-001/Data2023/BW0516/"
    #folder = "brewton"
    #mg_folder = "/home/lixin/Classes/Fall22Lab/Data2023-20230714T162453Z-001/Data2023/EV0511/"
    #folder = "tallassee"
    img_folder = "/home/lixin/Classes/Fall22Lab/Data2023-20230714T162453Z-001/Data2023/EV0519/"
    folder = "ev0519"
    for image in os.listdir(img_folder):
        if image.endswith(".jpg"):
            img_dirs.append(image)
    img_dirs = np.sort(img_dirs)
    if len(sys.argv) > 1:
        img_select = int(sys.argv[1])-1
    else:
        img_select = None

    interpreter = init_model(model_path)
    input_details = interpreter.get_input_details()
    # Get Width and Height
    input_shape = input_details[0]['shape']
    height = input_shape[1]
    width = input_shape[2]
    # Get input index
    input_index = input_details[0]['index']

    results = []
    if img_select is None:
        for i, img_file in enumerate(img_dirs):
            frame = cv2.imread(img_folder + img_file)
            print(img_file)
            if (frame.shape[0] < frame.shape[1]):
                frame = cv2.rotate(frame,cv2.ROTATE_90_CLOCKWISE)
            count, area = GeneratOutput(frame, interpreter, input_index, input_shape, 0.7)
            results.append([img_file, count, area])
        np.savetxt("/home/lixin/Classes/Fall22Lab/ml.csv",np.array(results),delimiter=',',header='file,count,area',fmt="%s")
    else:
        frame = cv2.imread(img_folder + img_dirs[img_select])
        if (frame.shape[0] < frame.shape[1]):
            frame = cv2.rotate(frame,cv2.ROTATE_90_CLOCKWISE)
        count, area = GeneratOutput(frame, interpreter, input_index, input_shape, 0.7)


    #print(resize_ratio)
    #cv2.putText(results,f'avg_area: {avg_area:.2f}',(0,50), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,0,255), thickness=2)
    
