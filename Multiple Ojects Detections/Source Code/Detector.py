import cv2

import numpy as np
import time


class Detector:
    def __init__(self,videoPath,configPath,modelPath,classesPath):
        self.videoPath = videoPath
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath
        
        
        self.net = cv2.dnn_DetectionModel(self.modelPath,self.configPath)
        self.net.setInputSize(320,320)
        self.net.setInputScale(1.0/127.5)
        self.net.setInputMean(127.5)
        self.net.setInputSwapRB(True)
        
        self.readClasses()
        
        
        
        
    def readClasses(self):
        with open(self.classesPath,'r') as f:
            self.classesList = f.read().splitlines()
        
        
        self.classesList.insert(0, '__Background__')
        #print(self.classesList)
        #print(len(self.classesList))
        self.colorList = np.random.uniform(low=0,high=255,size=(len(self.classesList),3))
        
    def onVideo(self):
        cap = cv2.VideoCapture(0)
        # add ='https://192.168.1.2:8080/video'
        if(cap.isOpened()==False):
            print("Error opening file")
            return
        # image= cap.open(add)
        # image= cap.read()
        (success,image) = cap.read()
        
        while True:
            classLabelsIDs,confidences,bboxs = self.net.detect(image,confThreshold = 0.5)
            bboxs = list(bboxs)
            confidences = list(np.array(confidences).reshape(1,-1)[0])
            
            bboxIdx = cv2.dnn.NMSBoxes(bboxs, confidences, score_threshold=0.5, nms_threshold=0.2)
            
            if len(bboxIdx) !=0:
                for i in range(0,len(bboxIdx)):
                    
                    bbox = bboxs[np.squeeze(bboxIdx[i])]
                    classConfidence = confidences[np.squeeze(bboxIdx[i])]
                    classlabelID = np.squeeze(classLabelsIDs[np.squeeze(bboxIdx[i])])
                    #print(classlabelID)
                    if classlabelID<80:
                        # print(classlabelID)
                        classLabel = self.classesList[classlabelID]
                        classColor = [int(c) for c in self.colorList[classlabelID]]
                        displayText = classLabel
                        x,y,w,h = bbox
                        
                        cv2.rectangle(image, (x,y),(x+w,y+h), color=classColor,thickness=1)
                        cv2.putText(image, displayText, (x,y-10), cv2.FONT_HERSHEY_PLAIN, 1, classColor)
                    
            cv2.imshow('result',image)
            
            key= cv2.waitKey(400) & 0xFF
            if key == ord('q'):
                break 
            (success,image) = cap.read()
            # image= cap.open(add)
        cv2.destroyAllWindows()
                
                
                
                
                
                
                
                
                
                
                
                