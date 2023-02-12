from Detector import *
import os
import cv2

import numpy as np
import time









#def main():
videoPath =0
configPath= 'D:/Private Projects/object detaction 2/object detaction 2/frozen_inference_graph.pb'
# modelPath = 'C:/Users/DELL/object detaction 2/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
modelPath = 'D:/Private Projects/object detaction 2/object detaction 2/object_detection.pbtxt'
classesPath = 'D:/Private Projects/object detaction 2/object detaction 2/coco.names.txt'
    
detector = Detector(videoPath,configPath,modelPath,classesPath)
detector.onVideo()
    
'''if __name__ == 'main':
    #main()
    print('ddd')'''