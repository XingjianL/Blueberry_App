# buildozer android debug deploy run
# buildozer android debug
# ./adb install .\myapp-0.1-arm64-v8a_armeabi-v7a-debug.apk
# ./adb logcat python:V org.test.myapp:V *:S


from random import random
import time
from kivy.uix.image import Image
from PIL import Image as pilImage
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.uix.gridlayout import GridLayout
from kivy.uix.tabbedpanel import TabbedPanel
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.clock import Clock
from kivy.utils import platform
from kivy.graphics.texture import Texture
from android.permissions import request_permissions, Permission, check_permission
from android.storage import app_storage_path, primary_external_storage_path, secondary_external_storage_path
import threading

import os
import cv2
import numpy as np

from blueberry import AppProcess
import ml.deep_learning_with_opencv as ml_code
import ml.deep_learning_with_tf as tf_code
class BlueBerry(Image):
    def run(self, frame, scaling = 1, min_r = 20, max_r = 55):
        print("run: ", frame.shape)
        try:
            self.output_images, self.counts, self.avg_areas, self.watersheds, self.misc_output = AppProcess.process_image(frame, scaling, min_r, max_r)
        except Exception as e:
            print(e)
            self.counts = None
            self.avg_areas = None
            self.misc_output = [None,None]
            print("run: no circles found")
            return None, None
        self.output_kivy = [self.cv2Kivy(img) for img in self.output_images]
        print(self.watersheds[0].shape)
        self.watershed_kivy = [self.cv2Kivy(cv2.cvtColor(np.uint8(np.dstack(img).T), cv2.COLOR_GRAY2BGR)) for img in self.watersheds]
        print(self.misc_output)
        return self.output_kivy + self.watershed_kivy, [self.counts[0], self.avg_areas[1]]
    # convert opencv images to kivy textures
    def cv2Kivy(self, image):
        print("cv2kv: ", image.shape)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #print(image.dtype)
        buf1 = cv2.flip(image,0)
        buf = buf1.tobytes()
        texture1 = Texture.create(size=(image.shape[1], image.shape[0]), colorfmt='rgb')
        texture1.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        return texture1
    
    def kivy2CV(self, texture):
        print("kv2CV:", texture)
        size = texture.size
        pixels = texture.pixels
        #print(size, pixels)
        image = np.array(pilImage.frombytes(mode='RGBA', size=size, data=pixels))
        image = cv2.cvtColor(image,cv2.COLOR_RGBA2BGR)
        #cv2.imshow("1",image)
        #cv2.waitKey()
        return image

    def load_dir(self, dir):
        images = []
        for image in os.listdir(dir):
            if image.endswith(".jpg"):
                images.append(image)
        print(images)
        return images
    def normalize_checker(self, frame):
        detected_check, factor = AppProcess.checkerboard(frame)
        return self.cv2Kivy(detected_check), factor
    pass

class BlueberryWidget(Widget):

    def on_touch_down(self, touch):
        color = (random(), 1, 1)
        with self.canvas:
            Color(*color, mode='bgr')
            d = 30.
            Ellipse(pos=(touch.x - d / 2, touch.y - d / 2), size=(d, d))
            touch.ud['line'] = Line(points=(touch.x, touch.y))

    def on_touch_move(self, touch):
        touch.ud['line'].points += [touch.x, touch.y]


#Builder.load_file("blueberry.kv")

class BlueberryApp(App):
    def check_permissions(self, perms):
        for perm in perms:
            if check_permission(perm) != True:
                return False
        return True
    def build(self):
        perms = [Permission.CAMERA, Permission.WRITE_EXTERNAL_STORAGE, Permission.READ_EXTERNAL_STORAGE]
        if self.check_permissions(perms)!= True:
            request_permissions(perms)    # get android permissions     
            exit()                        # app has to be restarted; permissions will work on 2nd start

        return Tab()

    

class Tab(TabbedPanel):
    def __init__(self, **kwargs):
        super(Tab, self).__init__(**kwargs)
        self.process = BlueBerry()
        self.img_id = -1
        self.image_names = []
        self.total_images = 0
        self.current_image_path = ''
        self.current_image = 0

        self.outputs = [0,0,0,0] # timestamp, method, counts, estimate

    def print_path(self,dir):
        self.img_dir = dir
        self.image_names = self.process.load_dir(dir)
        self.total_images = len(self.image_names)
        print(dir)

    def process_image(self):
        self.current_outputs = self.process.run(cv2.imread(self.current_image_path))
    
    def process_all(self):
        threading.Thread(target=self.process_all_image).start()

    def process_all_image(self):
        self.ids.process_all.disabled = True
        self.img_id = -1
        for i in range(self.total_images-1):
            self.next_image()
            self.display_input_label()
            self.process_image()
        self.ids.process_all.disabled = False

    def next_image(self):
        self.img_id += 1
        if (self.img_id > self.total_images-1):
            self.img_id = 0
        #self.i1.texture = self.process.cv2Kivy(cv2.imread(self.img_dir + '/' + self.image_names[self.img_id]))
        self.current_image_path = self.img_dir + '/' + self.image_names[self.img_id]
        self.ids.i1.source = self.current_image_path
        #print(self.current_image_path)
        return self.current_image_path
        

    def previous_image(self):
        self.img_id -= 1
        #self.i1.texture = self.process.cv2Kivy(cv2.imread(self.img_dir + '/' + self.image_names[self.img_id]))
        if (self.img_id < 0):
            self.img_id = 51
        self.current_image_path = self.img_dir + '/' + self.image_names[self.img_id]
        self.ids.i1.source = self.current_image_path
        return self.current_image_path
    
    #######
    #   Output Display
    #######

    def display_output(self):
        
        if self.current_outputs is None:
            temp = self.current_image.copy()
            cv2.circle(temp,(100,100),int(self.ids.slider_max_r.value),color=(0,255,0),thickness=-1)
            cv2.circle(temp,(100,100),int(self.ids.slider_min_r.value),color=(0,0,255),thickness=-1)
            edit_photo = self.process.cv2Kivy(temp)
            blank = self.process.cv2Kivy(np.zeros((100,100,3),dtype=np.uint8))
            photo = self.process.cv2Kivy(self.current_image)
            print("display_output: ", self.current_image.shape, photo)
            print("display_output: ", self.current_image[10,10,:])
            return photo, edit_photo, blank, blank, blank
        else:
            temp = self.process.output_images[0].copy()
            cv2.circle(temp,(100,100),int(self.ids.slider_max_r.value),color=(0,255,0),thickness=-1)
            cv2.circle(temp,(100,100),int(self.ids.slider_min_r.value),color=(0,0,255),thickness=-1)
            edit_photo = self.process.cv2Kivy(temp)
            return edit_photo, self.current_outputs[1], self.current_outputs[5], self.current_outputs[3], self.current_outputs[6]

    def display_input_label(self):
        inputtext = self.current_image_path
        inputtext += "\n imageID: " + str(self.img_id)
        inputtext += "\n total: " + str(self.total_images)
        print(self.ids.input_label.text)
        self.ids.input_label.text = inputtext
        pass

    ######
    #   Camera
    ######

    def capture(self):
        camera = self.ids.camera
        print("capture: ", camera.index)
        #print(test.isOpened())
        timestr = time.strftime("%Y%m%d_%H%M%S")
        self.current_image = self.process.kivy2CV(camera.texture)
        self.current_image = cv2.rotate(self.current_image, cv2.ROTATE_90_CLOCKWISE)
        self.current_image = cv2.flip(self.current_image, 1)
        print("capture: ",self.current_image.shape)
        self.outputs[0] = timestr
        #camera.export_to_png("IMG_{}.png".format(timestr))
    
    def cam_process(self):
        start = time.time()
        print("process: ",self.current_image.shape)
        self.current_outputs, outputs = self.process.run(self.current_image, scaling = 1, min_r = int(self.ids.slider_min_r.value), max_r = int(self.ids.slider_max_r.value))
        if outputs is not None:
            self.outputs[1] = "HT"
            self.outputs[2] = outputs[0]
            self.outputs[3] = outputs[1]
        end = time.time()
        print(f"Processing took: {end-start} seconds")

    def normalize_image(self):
        texture, factor = self.process.normalize_checker(self.current_image)
        if factor is None:
            return texture
        self.current_image = cv2.resize(self.current_image, (0,0), fx=factor,fy=factor)
        return texture
    ######
    #   ML classification
    ######
    #def output_ml_classification(self):
#
    #    label_path = 'ml/synset_words.txt'
    #    prototxt_path = 'ml/bvlc_googlenet.prototxt'
    #    model_path = 'ml/bvlc_googlenet.caffemodel'
    #    image = self.current_image.copy()
#
    #    out_img = ml_code.run_ml_caffe(image, model_path, prototxt_path, label_path)
    #    out_texture = self.process.cv2Kivy(out_img)
    #    return out_texture

    def output_ml_yolo(self):
        classes_path = "ml/coco.names"
        cfg_path = 'ml/yolov3-tiny.cfg'
        weight_path = 'ml/yolov3-tiny.weights'

        image = self.current_image.copy()
        out_img = ml_code.run_ml_yolo(image, weight_path, cfg_path, classes_path)
        out_texture = self.process.cv2Kivy(out_img)
        return out_texture
    def output_ml_yoloonnx(self):
        weight_path = 'ml/best.onnx'

        image = self.current_image.copy()
        out_img = ml_code.run_ml_yoloonnx(image, weight_path)
        out_texture = self.process.cv2Kivy(out_img)
        return out_texture
    def output_ml_yolotfliten(self):
        start = time.time()
        weight_path = 'ml/yolov5n-best-fp16.tflite'
        image = self.current_image.copy()
        out_img, count, avg_area = tf_code.run_ml_tflite(image, weight_path)
        self.outputs[1] = "ML_Nano"
        self.outputs[2] = count
        self.outputs[3] = avg_area
        out_texture = self.process.cv2Kivy(out_img)
        end = time.time()
        print(f"Processing took: {end-start} seconds")
        return out_texture
    def output_ml_yolotflites(self):
        start = time.time()
        weight_path = 'ml/yolov5s-best-fp16.tflite'
        image = self.current_image.copy()
        out_img, count, avg_area = tf_code.run_ml_tflite(image, weight_path)
        self.outputs[1] = "ML_Small"
        self.outputs[2] = count
        self.outputs[3] = avg_area
        out_texture = self.process.cv2Kivy(out_img)
        end = time.time()
        print(f"Processing took: {end-start} seconds")
        return out_texture
    def output_ml_yolotflitem(self):
        start = time.time()
        weight_path = 'ml/yolov5m-best-fp16.tflite'
        image = self.current_image.copy()
        out_img, count, avg_area = tf_code.run_ml_tflite(image, weight_path)
        self.outputs[1] = "ML_Medium"
        self.outputs[2] = count
        self.outputs[3] = avg_area
        out_texture = self.process.cv2Kivy(out_img)
        end = time.time()
        print(f"Processing took: {end-start} seconds")
        return out_texture
    #def output_ml_yolotflite4(self):
    #    weight_path = 'ml/yolov5l-best-fp16.tflite'
    #    image = self.current_image.copy()
    #    out_img = tf_code.run_ml_tflite(image, weight_path)
    #    out_texture = self.process.cv2Kivy(out_img)
    #    return out_texture
    ######
    #   Config
    ######

    def config(self, sc, so, ap, dm):
        print(sc, so, ap, dm)
    def saveCSV(self):
        testfile = 'bytes([1,2,3,4])'
        if platform =='android':
            csvFileName = os.path.join(primary_external_storage_path(),'Download/testfile.csv')
            print(csvFileName)
            with open(csvFileName, 'a') as f:
                f.write('\n'+str(self.outputs[0])+','+str(self.outputs[1])+','+str(self.outputs[2])+','+str(self.outputs[3]))
                self.outputs = [self.outputs[0], 0, 0, 0]
                f.close()
        print(f"writing to {csvFileName}")
    pass



if __name__ == '__main__':
    BlueberryApp().run()