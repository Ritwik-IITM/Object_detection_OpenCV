
import cv2
import matplotlib.pyplot as plt
%matplotlib inline
import os
import glob

config_file='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model='frozen_inference_graph.pb'
model=cv2.dnn_DetectionModel(frozen_model,config_file)

classlabels=[]
labels='labels.txt'
with open(labels,'rt') as label:
    classlabels=label.read().rstrip('\n').split('\n')

print(classlabels)

model.setInputSize(320,320)
model.setInputScale(1/127.5)
model.setInputMean(127.5)
model.setInputSwapRB(True)

# Function to detect object in image
def object_detection(image):
    img=cv2.imread(image)
    print(image)
    print('*'*50)
    classIndex,confidence,bbox=model.detect(img,confThreshold=0.5)
    index_list=list(classIndex)
    for index in index_list:
        print("Index: "+str(index),classlabels[index-1])
    print('*'*50)
    for classindex,conf,boxes in zip(classIndex.flatten(),confidence.flatten(),bbox):
        cv2.rectangle(img,boxes,(0,0,255),5)
        cv2.putText(img,classlabels[classindex-1],(boxes[0]+10,boxes[1]+10),cv2.FONT_HERSHEY_SIMPLEX,3,(255,0,0),3)
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

# passing the image individually
object_detection('robot.jpg')
object_detection('people-wine.jpg')
object_detection('red-traffic-light-and-stop-sign.jpg')

# Reading the all '.jpg' files in folder at a time
# for file in os.listdir(r'D:\Projects\Object_detection\Using_OpenCV'):
#     if file.endswith('.jpg'):
#         object_detection(file)


# Object detection from video
# Video

def object_detection_video(video):
  cap=cv2.VideoCapture(video)
  if not cap.isOpened():
      cap=cv2.VideoCapture(0)
  if not cap.isOpened():
      raise IOerror('cant open the file')

  while True:
      re,frame=cap.read()
      classIndex,confidence,bbox=model.detect(frame,confThreshold=0.5)
      index_list=list(classIndex)
      for index in index_list:
          print("Index: "+str(index),classlabels[index-1])
      print('*'*50)
      for classindex,conf,boxes in zip(classIndex.flatten(),confidence.flatten(),bbox):
          cv2.rectangle(frame,boxes,(0,0,255),5)
          cv2.putText(frame,classlabels[classindex-1],(boxes[0]+10,boxes[1]+10),cv2.FONT_HERSHEY_SIMPLEX,3,(255,0,0),3)
      cv2.imshow('Object Detection',frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break
  cap.release()
  cv2.destroyAllWindows()
  
object_detection_video('pexels-mike-bird-2103099-3840x2160-60fps.mp4')
