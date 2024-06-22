import cv2, os
#creating database
#captures images
haar_file = 'D:/Open CV/Lesson 8/haarcascade_frontalface_default.xml'

#all the faces data will be
#present folder
datasets = 'D:/Open CV/Lesson 8/datasets'

#sub data sets of folder
sub_data = 'josh'

path = os.path.join(datasets, sub_data)
if not os.path.isdir(path):
    os.mkdir(path)#
#width, height of img
(width, height) = (130,100)

face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)

count = 1
while count <= 30:
    (ret, im) = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #face_cascade.detectMultiScale(image, scaleFactor, Min Neighbours)
    # Parameter specifying how many neighbors each candidate rectangle should have to retain it. 
    #This parameter will affect the quality of the detected faces: 
    #higher value results in less detections but with higher quality.
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    for (x,y,w,h)in faces:
        cv2.rectangle(im, (x,y),(x + w, y + h), (255,0,0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width,height))
        cv2.imwrite('% s/% s.png' % (path, count), face_resize)
    count += 1

    cv2.imshow('OpenCV', im)
    key = cv2.waitKey(0)
    if key == 27:
        break
