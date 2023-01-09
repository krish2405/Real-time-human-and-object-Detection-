from cProfile import label
from fileinput import filename
import cv2
from matplotlib.cbook import flatten
import matplotlib.pyplot as plt


config_file='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model='frozen_inference_graph.pb'


model=cv2.dnn_DetectionModel(frozen_model,config_file)

classLabels=[]
filename='labels.txt'
with open(filename,'rt') as fpt:
    classLabels=fpt.read().rstrip('\n').split('\n')


# print(classLabels)

# print(len(classLabels))


# read an image

model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5,))
model.setInputSwapRB(True)

img =cv2.imread('test1.jpg')

plt.imshow(img)
plt.show()

plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.show()

ClassIndex,confidence,bbox=model.detect(img,confThreshold=0.5)
print(ClassIndex)

font_scale=4
font=cv2.FONT_HERSHEY_PLAIN

for ClassInd, conf,boxes in zip(ClassIndex.flatten(),confidence.flatten(),bbox):
    cv2.rectangle(img,boxes,(255,0,0),5)
    cv2.putText(img,classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40),font,fontScale=font_scale,color=(0,255,0),thickness=4)

plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.show()


cap=cv2.VideoCapture(0)

if not cap.isOpened():
    cap=cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError('cannot open video')


font_scale=3
font=cv2.FONT_HERSHEY_PLAIN



while True:
    ret,frame=cap.read()

    # print(frame)

    ClassIndex,confidence,bbox=model.detect(frame,confThreshold=0.5)

    print(ClassIndex)
    if (len(ClassIndex)!=0):
        for ClassInd, conf,boxes in zip(ClassIndex.flatten(),confidence.flatten(),bbox):
            if(ClassInd<=80):
                 cv2.rectangle(frame,boxes,(0,0,255),5)
                 cv2.putText(frame,classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40),font,fontScale=font_scale,color=(0,255,0),thickness=4)

    cv2.imshow('Object Detection Tutorial',frame)

    if cv2.waitKey(2) & 0xFF ==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

