# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 20:49:10 2018

@author: Jingyi Zou (300009753)
"""

from scipy import misc 
import tensorflow as tf 
import detect_face 
import cv2 
import matplotlib.pyplot as plt 
import os
  
minsize = 20 # minimum size of face 
threshold = [ 0.6, 0.7, 0.7 ] # three steps's threshold 
factor = 0.709 # scale factor 
gpu_memory_fraction=1.0
  
  
print('Creating networks and loading parameters') 
  
with tf.Graph().as_default(): 
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction) 
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) 
    with sess.as_default(): 
      pnet, rnet, onet = detect_face.create_mtcnn(sess, None) 

'''
Here choose one test image.
'''  
#image_path = os.path.join(os.getcwd(),'NBA_4.jpg') 
image_path = os.path.join(os.getcwd(),'UO_02.jpg')   
  
img = misc.imread(image_path)       
bounding_boxes, landmarks = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor) 
nrof_faces = bounding_boxes.shape[0]

print('The number of faces we find：{}'.format(nrof_faces))
print('The dimension of bounding box:', bounding_boxes.shape)
print('The dimension of landmarks:', landmarks.shape)   

print(bounding_boxes[3][4])
print(landmarks) 
print(len(bounding_boxes))

crop_faces=[] 
for i in range(bounding_boxes.shape[0]): 
  face_position=bounding_boxes[i].astype(int)
  print(face_position[0:4]) 
  cv2.rectangle(img, (face_position[0], face_position[1]), (face_position[2], face_position[3]), (0, 255, 0), 2) 
  for j in range(landmarks.shape[1]):
      cv2.circle(img, (int(landmarks[0][j]),int(landmarks[5][j])), 1, (0,255,0), 4)
      cv2.circle(img, (int(landmarks[1][j]),int(landmarks[6][j])), 1, (0,255,0), 4)
      cv2.circle(img, (int(landmarks[2][j]),int(landmarks[7][j])), 1, (0,255,0), 4)
      cv2.circle(img, (int(landmarks[3][j]),int(landmarks[8][j])), 1, (0,255,0), 4)
      cv2.circle(img, (int(landmarks[4][j]),int(landmarks[9][j])), 1, (0,255,0), 4)
  crop=img[face_position[1]:face_position[3], 
       face_position[0]:face_position[2],] 
    
  crop = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_CUBIC ) 
  print(crop.shape) 
  crop_faces.append(crop) 
  plt.imshow(crop) 
  plt.show() 
 
plt.imshow(img) 
plt.show()