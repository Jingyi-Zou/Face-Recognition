#encoding=utf-8
from scipy import misc
from PIL import Image
import tensorflow as tf
import numpy as np
import facenet
import align.detect_face
import os
import shutil
import types


model = "model/20170512-110547.pb"
base_img_path = "pics/anchors"
save_path = "pics/resize_anchors"
image_size = 160
margin = 44
gpu_memory_fraction = 1.0


def load_imgs(img_path = base_img_path,use_to_save = True):
    minsize = 20
    threshold = [0.6,0.7,0.7]
    factor = 0.709
    result = {}
    print('Creating networks and loading parameters')

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
            
    img_paths = os.listdir(img_path)
    #return names of files in the designated file folder, construct a list in the Alphabetical order
    for image in img_paths:       
        if image == '.DS_Store':
            continue        
        aligned = mtcnn(os.path.join(img_path, image),minsize,pnet,rnet,onet,threshold,factor) 
        #if no face detected in the image, remove it 
        if aligned is None:
            img_paths.remove(image)
            continue
        if use_to_save: #
            result[image.split('.')[0]] = aligned
        else:
            prewhitened = facenet.prewhiten(aligned)  
            result[image.split('.')[0]] = prewhitened
    return result


def mtcnn(img_path,minsize, pnet, rnet, onet, threshold,factor):
    img = misc.imread(img_path, mode='RGB')  # convert images to matrix
    img_size = np.asarray(img.shape)[0:2] #convert list to array, contains size and channel amount
    #face detection
    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold,
                                                      factor)  
    # find four vertexes to build bounding box
    if len(bounding_boxes) < 1:
        print("can't detect face, remove ", img_path)  # if no face has been detected, remove the picture
        return None
    det = np.squeeze(bounding_boxes[0, 0:4])
    # add boundary for face
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[0] - margin / 2, 0)
    bb[1] = np.maximum(det[1] - margin / 2, 0)
    bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
    bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
    # get cropped face
    cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
    # resize the face to the suitable model-input size
    aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
    return aligned


def save_imgs():
    result = load_imgs()
    print("load pic end. result size: %d" %len(result))
    for image_path in result:
        #build path for image-saving
        img = Image.fromarray(result[image_path].astype('uint8'))
        img_path = os.path.join(save_path, image_path.split('/')[-1] + ".jpg")
        img.save(img_path)
    print("save end")


def embedPic(pics):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            facenet.load_model(model)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")  # convert image to 128-dimension face embedding 
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            feed_dict = {images_placeholder:pics,phase_train_placeholder:False}
            emb = sess.run(embeddings,feed_dict=feed_dict)
            return emb


def check_all_in_database(img_path):
    unembed = []
    database_labels = os.listdir(base_img_path)
    if type(img_path) == list:
        for image_path in img_path:
            image = image_path.split('/')[-1]  
            if image in database_labels:
                print("image in database:%s", image)
            else:
                unembed.append(image_path)
                shutil.copyfile(image_path, os.path.join(base_img_path, image_path.split("/")[-1]))
    elif os.path.isfile(img_path):
        image = img_path.split('/')[-1]
        if image in database_labels:
            print("image in database:%s", image)
        else:
            unembed.append(img_path)
            shutil.copyfile(img_path,os.path.join(base_img_path,img_path.split("/")[-1]))
    elif os.path.isdir(img_path):
        test_labels = os.listdir(img_path)
        for label in test_labels:
            if label == '.DS_Store':
                continue
            if label in database_labels:
                continue
            # save file
            unembed.append(os.path.join(img_path,label))
            shutil.copyfile(os.path.join(img_path,label), os.path.join(base_img_path, label))
                
    if len(unembed) == 0:
        print("all data in the database")
        return

    print("datas not in the database:")
    print(unembed)
    minsize = 20
    threshold = [0.6,0.7,0.7]
    factor = 0.709
    print("start emb...")
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
            imgs = []
            labels = []
            for image in unembed:
                aligned = mtcnn(image,minsize,pnet,rnet,onet,threshold,factor)
                if aligned is None:
                    unembed.remove(image)
                    continue
                prewhitened = facenet.prewhiten(aligned)
                imgs.append(prewhitened)
                labels.append(image.split('/')[-1].split('.')[0])
            emb = embedPic(imgs)
            print("emb finish")
            print("start save...")
            np_sort = np.column_stack([np.array(labels),emb])
            current_emb = np.load("model/database.npy")
            current_emb = np.row_stack([current_emb,np_sort])
            np.save("model/database.npy", current_emb)
            save_imgs()
            print("save finish")










