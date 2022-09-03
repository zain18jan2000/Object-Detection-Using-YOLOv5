import torch
import cv2

model = torch.hub.load('yolov5', 'yolov5n', source= 'local')

cap = cv2.VideoCapture('video.mp4')

while True:
    img = cap.read()[1]
    if img is None:
        break
    result = model(img)
    df = result.pandas().xyxy[0]

    for ind in df.index:
        x1, y1 = int(df['xmin'][ind]), int(df['ymin'][ind])
        x2, y2 = int(df['xmax'][ind]), int(df['ymax'][ind])
        label = df['name'][ind]
        conf = df['confidence'][ind]
        text = label + ' ' + str(conf.round(decimals= 2))
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)

    cv2.imshow('Video',img)
    cv2.waitKey(10)

'''
img = cv2.imread('test.jpg')
img = cv2.resize(img,(1000,650))
result = model(img)
print(result)
df = result.pandas().xyxy[0]
print(df)

for ind in df.index:
    x1, y1 = int(df['xmin'][ind]), int(df['ymin'][ind])
    x2, y2 = int(df['xmax'][ind]), int(df['ymax'][ind])
    label = df['name'][ind]
    cv2.rectangle(img,(x1,y1), (x2,y2), (255,255,0),2)
    cv2.putText(img,label,(x1,y1-5), cv2.FONT_HERSHEY_PLAIN, 2,(255,255,0),2)

cv2.imshow('IMAGE',img)
cv2.waitKey(0)
'''