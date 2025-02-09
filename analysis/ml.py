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
    #print(image.shape)
    #plt.imshow(input_data[0])
    #plt.show()
    interpreter.set_tensor(input_index,input_data)
    interpreter.invoke()
    
    output_details = interpreter.get_output_details()
    out = np.squeeze(interpreter.get_tensor(output_details[0]['index']))
    #classes = np.squeeze(interpreter.get_tensor(output_details[1]['index']))
    #scores = np.squeeze(interpreter.get_tensor(output_details[2]['index']))
    #print(out.shape)
    out = out.T
    boxes = []
    confidences = []
    classIDs = []
    radii = []
    radii_center = []
    imgh,imgw = image.shape[:2]
    for detection in out:
        confidence = detection[4]
        if confidence > conf:
            #print(detection[:4])
            #scores = detection[5:]
            #classID = np.argmax(scores)
            #if scores[classID] > conf:
                #print(detection)
            confidences.append(float(confidence))
            #classIDs.append(classID)
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
            radii_center.append((x+0.5*w)/imgw)
            radii_center.append((y+0.5*h)/imgh)
            #text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
            #cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    plt.imshow(image)
    plt.show()
    #print("tests: ", image.shape)
    #print("tests: ", np.average(radii))
    #image = image[:,80:-80,:]
    return image, np.array(radii), radii_center
def GeneratOutput(frame, interpreter, input_index, input_shape, conf = 0.5):
    frame_size = frame.shape[:2]
    max_side = np.argmax(frame_size)
    #print(max_side,frame_size[max_side])
    resize_ratio = input_shape[max_side+1]/frame_size[max_side]
    new_shape = (int(frame_size[1]*resize_ratio), int(frame_size[0]*resize_ratio))
    frame = cv2.resize(frame, new_shape)
    if (frame.shape[0] < frame.shape[1]):
        #frame = cv2.rotate(frame,cv2.ROTATE_90_CLOCKWISE)
        frame = cv2.copyMakeBorder(frame,80,80,0,0,cv2.BORDER_REPLICATE)
    else:
        frame = cv2.copyMakeBorder(frame,0,0,80,80,cv2.BORDER_REPLICATE)

    #print(frame.shape)
    results, radii, radii_center = run_ml_tflite(frame,interpreter, input_index,conf)

    # used for results (found wrong after the results were recorded, corrected in excel sheet)
    #avg_area = np.mean(np.pi * np.power(radii,2))/resize_ratio 

    areas = np.pi * np.power(radii/resize_ratio,2)

    # correct pixel area at original 4032x3024 image
    avg_area = np.mean(areas)

    # variance of the area
    variance = np.var(areas)


    print(input_shape, frame_size[max_side])
    print("\taverage area (see code comments): ", avg_area)
    print("\tcount: ", len(radii))
    # plt.imshow(results)
    # plt.show()
    #plt.close()
    return len(radii), avg_area , variance, areas, radii_center
if __name__ == "__main__":
    
    model_path = '/home/lixin/Classes/Fall22Lab/kivyapp/ml/models/v5m-best-fp16.tflite'
    #model_path = '/home/lixin/Classes/Fall22Lab/kivyapp/ml/models/v8m-best-fp16.tflite'
    #model_path = '/home/lixin/Classes/Fall22Lab/kivyapp/ml/models/v11m-best-fp16.tflite'
    #model_path = '/home/lixin/Classes/Fall22Lab/annotated_blueberry/yolo_trainings/5/runs/detect/gh-bi/weights/best_saved_model/best_float16.tflite'
    #model_path = '/home/lixin/Classes/Fall22Lab/annotated_blueberry/yolo_trainings/5/runs/detect/n/weights/best_saved_model/best_float16.tflite'
    #model_path = '/home/lixin/Classes/Fall22Lab/annotated_blueberry/yolo_trainings/5/runs/detect/s/weights/best_saved_model/best_float16.tflite'

    img_folders = [
        {"imgs":"/home/lixin/Classes/Fall22Lab/drive-download-20220904T003514Z-001/",
         "name":"auburn",
         "size_adjust":1662/15},
        {"imgs":"/home/lixin/Classes/Fall22Lab/Fairhope_052322/",
         "name":"fairhope",
         "size_adjust":1471/15},
         {"imgs":"/home/lixin/Classes/Fall22Lab/Data2023-20230714T162453Z-001/Data2023/EV0511/",
         "name":"tallassee",
         "size_adjust":1740/15},
        {"imgs":"/home/lixin/Classes/Fall22Lab/Data2023-20230714T162453Z-001/Data2023/BW0516/",
         "name":"brewton",
         "size_adjust":1678/15},
        {"imgs":"/home/lixin/Classes/Fall22Lab/Data2023-20230714T162453Z-001/Data2023/EV0519/",
         "name":"ev0519",
         "size_adjust":1740/15}
    ]
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
    for img_set in img_folders:
        img_dirs=[]
        for image in os.listdir(img_set["imgs"]):
            if image.endswith(".jpg"):
                img_dirs.append(image)
        img_dirs = np.sort(img_dirs)
        if img_select is None:
            for i, img_file in enumerate(img_dirs):
                frame = cv2.imread(img_set["imgs"] + img_file)
                print(img_file)
                #if (frame.shape[0] < frame.shape[1]):
                    #frame = cv2.rotate(frame,cv2.ROTATE_90_CLOCKWISE)
                count, area, variance, individual_areas, radii_center = GeneratOutput(frame, interpreter, input_index, input_shape, 0.5)
                if "PXL_20220621" in img_file and img_set["name"] == "auburn":
                    img_set["size_adjust"] =1555/15
                area = area / img_set["size_adjust"]**2
                individual_areas /= img_set["size_adjust"]**2
                results.append([img_file, count, area, f"\"{list(individual_areas)}\"", f"\"{list(radii_center)}\""])
            
        # else:
        #     frame = cv2.imread(img_set["imgs"] + img_dirs[img_select])
        #     print(img_dirs[img_select])
        #     if (frame.shape[0] < frame.shape[1]):
        #         frame = cv2.rotate(frame,cv2.ROTATE_90_CLOCKWISE)
        #     count, area, variance, individual_areas = GeneratOutput(frame, interpreter, input_index, input_shape, 0.5)
    np.savetxt("/home/lixin/Classes/Fall22Lab/ml_v5m_50.csv",np.array(results),delimiter=',',header='file_name,esti_count,esti_area,esti_individual_area,radii_centers',fmt="%s",comments="")

    #print(resize_ratio)
    #cv2.putText(results,f'avg_area: {avg_area:.2f}',(0,50), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,0,255), thickness=2)
    
