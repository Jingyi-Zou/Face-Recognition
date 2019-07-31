#encoding=utf-8
from __future__ import division, print_function, unicode_literals

import numpy as np
import os
import cv2
import random
import time

np.random.seed(42)

import tensorflow as tf
import numpy as np
import os
import json
import preProcess_mtcnn
import types
import functools

model = "model/20170512-110547.pb"
database_file = "model/database.npy"
model = "style_recongnization-master/facenet/model/20170512-110547.pb"

class SimpleData(object):
    key = 0.0
    value = 0.0
    target = 0.0
    def __init__(self):
        self.key = 0.0
        self.value = 0.0

    def __init__(self,key,value):
        self.key = key
        self.value = value

    def __init__(self,key,value,target):
        self.key = key
        self.value = value
        self.target = target


def data_2_json(obj):
    return {"key":obj.key,
            "value":obj.value,
            "target":obj.target}


def main(test_path,top_n = 100):

    function_initial='empty'
    function_collection='Z'
    function_recognition='X'
    function_exit='C'
    function_choose=function_initial
    f=open('E:/UOttawa/2018 Fall/ELG 5900 project/style_recongnization-master/facenet/model/information.txt','r')
    information_dict=json.loads(f.read())
    f.close()
    while(function_choose!=function_exit):   
            function_choose=input("Please choose the function(Z-collection; X-recognition,C-exit):")
            #CameraControl (function_choose)
            if function_choose==function_collection:
                i=0
                name,initial=input("Please input your name and initials of your first and last name(no middle name):").split(',',1)
                Birth=input("Please input your birthday(ex.19950401):")
                Stu_Num=input("Please input your student number(ex.300005965):")
                first=str(ord(initial[0]))
                last=str(ord(initial[1]))
                identity_key=first+last+Birth
                identity_value=' Name: '+name+'   Birthday: '+Birth+'   Student number: '+Stu_Num
                information_dict[identity_key]=identity_value
                f1=open('E:/UOttawa/2018 Fall/ELG 5900 project/style_recongnization-master/facenet/model/information.txt','w')
                f1.write(json.dumps(information_dict))
                f1.close()
                print("Please wait for the camera to open...")
                cap = cv2.VideoCapture(0)
                print("Please look straight to the camera, keep still and follow the instructions")
                while(i!=7):
                    ret, frame = cap.read()
                    k=cv2.waitKey(1)
                    cv2.imshow("FaceNet", frame)
                    print
                    if k & 0xFF == ord('q'):           
                        i=i+1
                        cv2.imwrite("E:/UOttawa/2018 Fall/ELG 5900 project/style_recongnization-master/facenet/pics/anchors/"+str(identity_key)+str(i)+".jpg", frame)
                        if i==3:
                            print("please turn left")
                        elif i==5:
                            print("please turn right")
                cap.release() 
                cv2.destroyAllWindows()
                print("Do face alignment and save the result")
                embedDatabaseAndSaveAsNpy()    
            elif function_choose==function_recognition:
                        print("Please wait for the camera to open...")
                        cap = cv2.VideoCapture(0)
                        print("Please look straight to the camera and keep still")
                        while(1):
                            ret, frame = cap.read()
                            cv2.imshow("FaceNet", frame)
                            print 
                            if cv2.waitKey(1) & 0xFF == ord('w'):
                                cv2.imwrite("E:/UOttawa/2018 Fall/ELG 5900 project/style_recongnization-master/facenet/imgs/123.jpg", frame)
                                break
                        cap.release() 
                        cv2.destroyAllWindows()

                        if os.path.exists(database_file) == False:
                            print("No database file exist, wrong operation!")
                            break
                        else:
                            preProcess_mtcnn.check_all_in_database(test_path)
                            current_emb = np.load(database_file)
                            sample_labels = load_labels(test_path) 
                            all_result = []
                            mDataDict = changeNpToDict(current_emb)
                        
                            for sample in sample_labels:                            
                                if sample in mDataDict:
                                    sample_emb = mDataDict[sample].astype('float64')
                                    result = []
                                    for compare_index in mDataDict:
                                        if compare_index == sample:
                                            continue
                                        compare_data = mDataDict[compare_index].astype('float64')
                                        dist = np.sqrt(np.sum(np.square(np.subtract(sample_emb,compare_data))))
                                        data = SimpleData(float(compare_index),dist,float(sample)) # calculate the Euclidean distance between two pictures
                                        result.append(data)
                                        all_result.append(data)
                        
                            all_result.sort(key = functools.cmp_to_key(cmp))  # reorder the Euclidean distance array
                            output = []
                            count = 0
                            for item in all_result:
                                for input_label in sample_labels:
                                    if float(item.key) == float(input_label):  
                                        continue
                                for exist_item in output:
                                    if exist_item.key == item.key:
                                        continue
                                item.key = str(item.key).split('.')[0] + ".jpg"
                                item.target = str(item.target).split('.')[0] + ".jpg"
                                output.append(item)
                                count = count + 1
                                if count == int(top_n):
                                    break
                            for item in output:
                                dist=float(item.value)
                                if dist>0.7:
                                    print('stranger')
                                else:
                                    get_key=item.key[0:12]
                                    print("picture:%s,     identity:%s,     distance:%s"%(item.target,item.key,item.value,))
                                    print('personal information:')
                                    print(information_dict[get_key])
                            data=np.load('E:/UOttawa/2018 Fall/ELG 5900 project/style_recongnization-master/facenet/model/database.npy')
                            Data=np.array(data)
                            hang=Data.shape[0]
                            Data_revised=np.delete(Data,hang-1,axis=0)
                            np.save('E:/UOttawa/2018 Fall/ELG 5900 project/style_recongnization-master/facenet/model/database.npy',Data_revised)
                            os.remove('E:/UOttawa/2018 Fall/ELG 5900 project/style_recongnization-master/facenet/imgs/123.jpg')
                            os.remove('E:/UOttawa/2018 Fall/ELG 5900 project/style_recongnization-master/facenet/pics/anchors/123.jpg')
                            os.remove('E:/UOttawa/2018 Fall/ELG 5900 project/style_recongnization-master/facenet/pics/resize_anchors/123.jpg')
                            return output
                        
                    
              


def cmp(data1, data2):
    if data1.value == data2.value:
        return 0
    if data1.value < data2.value:
        return -1
    if data1.value > data2.value:
        return 1
    
    
def changeDictToArray(dict):

    img_index = []
    data_content = []

    for index in dict:
        img_index.append(index)
        data_content.append(dict[index])

    return np.array(img_index),np.array(data_content)


def changeNpToDict(emb):
    result = {}
    for col in emb:
        result[col[0]] = np.delete(col,0,axis=0)
    return result


def load_labels(img_paths):
    output = []
    if type(img_paths) == list:
        for label in img_paths:
            output.append(label.split('/')[-1].split('.')[0])
    elif os.path.isfile(img_paths):
        output.append(img_paths.split('/')[-1].split('.')[0])
    else:
        labels = os.listdir(img_paths)
        for label in labels:
            if label == '.DS_Store':
                continue
            output.append(label.split('.')[0])
    return np.array(output)


# update the database.npy
def embedDatabaseAndSaveAsNpy():
    print("start load imgs...")
    imgs_dict = preProcess_mtcnn.load_imgs(use_to_save=False)
    img_index,img_content = changeDictToArray(imgs_dict)
    print("img loaded")
    print("start embedding...")
    img_emb = preProcess_mtcnn.embedPic(img_content)
    print("img emded")
    print("start saving...")
    np_soft = np.column_stack([img_index,img_emb[0:len(img_index)]])
    np.save("model/database.npy",np_soft)
    print("saved")
    preProcess_mtcnn.save_imgs()



if __name__ == '__main__':
    imgs = ["imgs/123.jpg"]
    result = main(imgs,1)