"""
将图像文件转换为矩阵类型文件，已知图像路径为'/root/cs'
"""

import face_recognition
import numpy as np
import os
from datetime import datetime
from flask import Flask, jsonify, request, redirect

# You can change this to any folder on your system
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'txt'}



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS




def load_faces(filePath, filePaths):
    """
    加载图片库
    :param filePath:图片存储路径
    :return: 所有图片组成的数组
    """
    if os.path.isdir(filePath):
        for file in os.listdir(filePath):
            file_abspath = filePath + '/' + file
            # print(file_abspath)
            if os.path.isdir(file_abspath):
                load_faces(file_abspath, filePaths)
            else:
                if allowed_file(file_abspath):
                    filePaths.append(file_abspath)
    else:
        if allowed_file(filePath):
            filePaths.append(filePath)
    return filePaths


def face_2_matrix():
    """
    图像转化为矩阵
    """
    num = 0
    for filePath in filePaths:
        path = filePath.split('/')[-1]
        try:
            obj = face_recognition.face_encodings(face_recognition.load_image_file(filePath))[0]
            absPath = '/root/cs/face_recognize/photoes_serialize/' + path
            np.savetxt(absPath, obj)
        except:
            continue
        print(num)
        num += 1

    pass


if __name__ == "__main__":
    filePath = '/root/cs/face_recognize/photoes'
    filePaths = []
    load_faces(filePath, filePaths)
    face_2_matrix()





