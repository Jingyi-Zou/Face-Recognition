# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 19:53:24 2018

@author: ncao0
"""
'''
import os
def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        print('root_dir:', root)  # 当前目录路径
        print('sub_dirs:', dirs)  # 当前路径下所有子目录
        print('files:', files)  # 当前路径下所有非目录子文件


file_name('E:/UOttawa/2018 Fall/ELG 5900 project/style_recongnization-master/webface test')
'''
from PIL import Image
import glob
import os
import time

#start = time.clock()
path='E:/UOttawa/2018 Fall/ELG 5900 project/style_recongnization-master backup/test series/test5'     
def file_name_walk(file_dir):
    for root, dirs, files in os.walk(file_dir):
        #print('root_dir:', root)  # 当前目录路径
        #print('sub_dirs:', dirs)  # 当前路径下所有子目录
        #print('files:', files)  # 当前路径下所有非目录子文件
        return files


#获取该目录下所有文件，存入列表中
f=os.listdir(path)
#print(len(f))
'''
i=0
file = open('E:/UOttawa/2018 Fall/ELG 5900 project/style_recongnization-master backup/test series/testlist.txt','w');
while(i!=100):
    f[i]='E:/UOttawa/2018 Fall/ELG 5900 project/style_recongnization-master backup/facenet/imgs2_adjustment/2.0 001/'+f[i]+'001.jpg'
    file.write(str(f[i]));
    file.write('\n')
    i=i+1
#print(f)
file.close();

f2=[]
file_2=open('E:/UOttawa/2018 Fall/ELG 5900 project/style_recongnization-master backup/test series/testlist.txt','r');
contents = file_2.readlines()
for name in contents:
    name = name.strip('\n')
    f2.append(name)
#end = time.clock()
#print(str(end-start))
#print(f2)

'''

i=0;
while(i!=(len(f))):
    path_sub='E:/UOttawa/2018 Fall/ELG 5900 project/style_recongnization-master backup/test series/test5/'+f[i]
    #print(f[i])
    f_sub=file_name_walk(path_sub)
    #print(f_sub)
    n=0
    while(n!=2):
        oldname=path_sub+'/'+f_sub[n]
        newname=path_sub+'/'+f[i]+f_sub[n]
        print(oldname,newname)
        os.rename(oldname,newname)
        im=Image.open(newname)
        newpath='E:/UOttawa/2018 Fall/ELG 5900 project/style_recongnization-master backup/facenet/imgs5/'+f[i]+f_sub[n]    
        im.save(newpath)
        n+=1
    #print(a)
    i=i+1
'''
j=0
while(j!=(len(f))):
    name1='E:/UOttawa/2018 Fall/ELG 5900 project/style_recongnization-master backup/facenet/imgs5/'+f[j]+'001.jpg'
    name2='E:/UOttawa/2018 Fall/ELG 5900 project/style_recongnization-master backup/facenet/imgs5/'+f[j]+'002.jpg'
    #name3='E:/UOttawa/2018 Fall/ELG 5900 project/style_recongnization-master backup/facenet/imgs5/'+f[i]+'003.jpg'
    savepath1='E:/UOttawa/2018 Fall/ELG 5900 project/style_recongnization-master backup/facenet/imgs5 001/'
    savepath2='E:/UOttawa/2018 Fall/ELG 5900 project/style_recongnization-master backup/facenet/imgs5 002/'
    im1=Image.open(name1)
    im1.save(savepath1+f[j]+'001.jpg')
    im2=Image.open(name2)
    im2.save(savepath2+f[j]+'002.jpg')
    #im3=Image.open(name3)
    #im3.save(savepath+f[i]+'003.jpg')
    j=j+1
'''