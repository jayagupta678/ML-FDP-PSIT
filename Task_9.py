import cv2
face_data = cv2.CascadeClassifier("C:\ProgramData\Anaconda3\Lib\site-packages\cv2\data\haarcascade_eye.xml")
image = cv2.imread(r"C:\Users\Ramendra\Desktop\fdp data\fdp 2017\img1.jpg", cv2.IMREAD_COLOR)

dataset = face_data.detectMultiScale(image)
for x,y,w,h in dataset:
    cv2.rectangle(image, (x,y), (x+w, y+h), (0,0,255), 2 )
cv2.imwrite(r'C:\Users\Ramendra\Desktop\fdp data\fdp 2017\result12.jpg', image)
