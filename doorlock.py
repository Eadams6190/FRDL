#id one image

import cv2
import os
import numpy
import faceRecognition as fr 
import sys

def main():
    print("aaaaaaaaaaaa")
    if len(sys.argv) < 2:
        print("Error: no image given")
    img = cv2.imread(sys.argv[1])
    print(sys.argv[1])
    print(type(img))
    faces_detected,gray_img = fr.faceDetection(img)
    print(faces_detected)
    print(type(gray_img))
    (x,y,w,h)=faces_detected[0]
    roi_gray=gray_img[y:y+h,x:x+h]
    facetester = fr.train_classifier([roi_gray],[1])
    if len(sys.argv) < 3:
        print("Error: no image given to compare")
    img2 = cv2.imread(sys.argv[2])
    print(sys.argv[2])
    print(type(img2))
    faces_detected2,gray_img2 = fr.faceDetection(img2)
    print(type(faces_detected2))
    (x,y,w,h)=faces_detected2[0]
    roi_gray2=gray_img2[y:y+h,x:x+h]
    label,confidence=facetester.predict(roi_gray2)#predicting the label of given image
    print("confidence:",confidence)
    print("label:",label)

if __name__ == '__main__':
    main()
