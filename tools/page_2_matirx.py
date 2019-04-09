"""
将图片转化为矩阵特征值进行存储
"""

import load_data
import face_recognition
import numpy as np
import os

def face_2_matrix():
    """
    将图片转化为矩阵类型，并提取人脸特征值，存储成*.out类型文件
    """
    num = 0
    for filePath in file_paths[0:10]:
        path = filePath.split('/')[-1].split('.')[0]+".out"
        
        obj = face_recognition.face_encodings(face_recognition.load_image_file(filePath))[0]
        absPath = '/root/cs/face_recognize/photoes_serialize/' + path
        print(absPath)
        if os.path.exists(absPath):
            print("该文件已存在")
            continue
        else:
            np.savetxt(absPath, obj)
        print(num)
        num += 1
    pass






if __name__ == '__main__':
    dir_path = '/root/cs/face_recognize/photoes'
    file_paths = []
    load_data.load_faces(dir_path,file_paths)
    print(len(file_paths))
    face_2_matrix()
