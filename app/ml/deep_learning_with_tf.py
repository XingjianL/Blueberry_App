import numpy as np
import time
import cv2
import math
from jnius import autoclass

File = autoclass('java.io.File')
FileIS = autoclass('java.io.FileInputStream')
FileC = autoclass('java.nio.channels.FileChannel')
FileCM = autoclass('java.nio.channels.FileChannel$MapMode')
FileM = autoclass('java.nio.MappedByteBuffer')
Interpreter = autoclass('org.tensorflow.lite.Interpreter')
InterpreterOptions = autoclass('org.tensorflow.lite.Interpreter$Options')
Tensor = autoclass('org.tensorflow.lite.Tensor')
DataType = autoclass('org.tensorflow.lite.DataType')
TensorBuffer = autoclass(
    'org.tensorflow.lite.support.tensorbuffer.TensorBuffer')
ByteBuffer = autoclass('java.nio.ByteBuffer')
FloatBuffer = autoclass('java.nio.FloatBuffer')
def run_ml_tflite(image, path):
    MODEL_INPUT_SIZE = 640
    model = File(path)
    model_file = FileIS(model)
    model_ch = model_file.getChannel()
    model_tf = model_ch.map(FileCM.READ_ONLY,0,model_ch.size())
    options = InterpreterOptions()
    options.setNumThreads(4)
    print(model.exists())
    print(model_file.read())
    print(model_tf.isLoaded())
    interpreter = Interpreter(model_tf,options)
    interpreter.allocateTensors()
    print(111)
    input_shape = interpreter.getInputTensor(0).shape()
    output_shape = interpreter.getOutputTensor(0).shape()
    output_type = interpreter.getOutputTensor(0).dataType()
    print(input_shape, output_shape, output_type)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.copyMakeBorder(img,0,0,80,80,cv2.BORDER_REPLICATE)
    #img = cv2.resize(img,(MODEL_INPUT_SIZE,MODEL_INPUT_SIZE))
    img_data = img/255.
    img_data = img_data.astype(np.float32)

    input = ByteBuffer.wrap(img_data.tobytes())
    output_tf = TensorBuffer.createFixedSize(output_shape,output_type)
    #input = FloatBuffer.allocate(interpreter.getInputTensor(0).numElements())
    print(interpreter.getInputTensor(0).numElements())


    start = time.time()
    interpreter.run(input, output_tf.getBuffer().rewind())

    end = time.time()
    print("[INFO] yolo took {:.5} seconds".format(end - start))
    print(output_tf)
    output = np.reshape(np.array(output_tf.getFloatArray()),output_shape)
    print(output.shape)
    print(np.max(output))
    #return np.reshape(np.array(output.getFloatArray()),output_shape)
    boxes = []
    confidences = []
    classIDs = []
    radii = []
    imgh,imgw = img_data.shape[:2] #############
    print(imgh,imgw)
    print("yolo: ", len(output))
    print(output[0].shape)
    for out in output:
    	#print(out.shape)
    	for detection in out:
            confidence = detection[4]
            if confidence > 0.5:
    			#print(detection[:4])
                scores = detection[5:]
                classID = np.argmax(scores)
                if scores[classID] > 0.5:
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
                    
    #print(len(boxes), boxes)
    #print(len(confidences))#, confidences)
    #print(len(classIDs))#, classIDs)
    if (len(boxes) == 0):
        return image, 0, 0
    indices = []
    #for box in enumerate(boxes):
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.2)
    #print(indices)
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            #color = [int(c) for c in colors[classIDs[i]]]
            cv2.rectangle(img, (x, y), (x + w, y + h), (255,255,255), 2)
            radii.append(np.min([w,h])/2)     # results
            #text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
            #cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    image = img[:,80:-80,:]
    avg_area = np.mean(np.pi * np.power(radii,2))
    count = len(radii)
    print(avg_area)
    return image, count, avg_area
    return None