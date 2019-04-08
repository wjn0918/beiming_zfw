"""
测试图像对比，已知图像路径为'/root/cs'
"""
# This is a _very simple_ example of a web service that recognizes faces in uploaded images.
# Upload an image file and it will check if the image contains a picture of Barack Obama.
# The result is returned as json. For example:
#
# $ curl -XPOST -F "file=@obama2.jpg" http://127.0.0.1:5001
#
# Returns:
#
# {
#  "face_found_in_image": true,
#  "is_picture_of_obama": true
# }
#
# This example is based on the Flask file upload example: http://flask.pocoo.org/docs/0.12/patterns/fileuploads/

# NOTE: This example requires flask to be installed! You can install it with pip:
# $ pip3 install flask

import face_recognition
import numpy as np
import os
from datetime import datetime
from flask import Flask, jsonify, request, redirect

# You can change this to any folder on your system
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    # Check if a valid image file was uploaded
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # The image file seems valid! Detect faces and return the result.
            return detect_faces_in_image(file)

    # If no valid image file was uploaded, show the file upload form:
    return '''
    <!doctype html>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
    <title>Is this a picture of Obama?</title>
    <h1>上传一张照片判断数据库中是否存在该人</h1>
    <form method="POST" enctype="multipart/form-data">
      <input type="file" name="file">
      <input type="submit" value="Upload">
    </form>
    '''

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







def detect_faces_in_image(file_stream):
    """
    图像对比
    :param filePath: 已知图像路径
    :param file_stream: 上传的图像
    :return: 是否存在
    """

    # Pre-calculated face encoding of Obama generated with face_recognition.face_encodings(img)


    # Load the uploaded image file
    img = face_recognition.load_image_file(file_stream)
    # Get face encodings for any faces in the uploaded image
    unknown_face_encodings = face_recognition.face_encodings(img)
    # print(type(unknown_face_encodings))
    face_found = False
    is_exists = False

    start_time = datetime.now()
    if len(unknown_face_encodings) > 0:
        face_found = True
        num = 0
        known_face_encoding = []
        num_people = len(filePaths)
        for known_face in filePaths:
            if num % 2 == 0:
                try:
                    known_face_encoding.append(np.loadtxt(known_face))
                except:
                    continue
                num += 1
                continue
            else:
                match_results = face_recognition.face_distance(known_face_encoding,unknown_face_encodings[0])
                for i in match_results:
                    if i < 0.6:
                        print(match_results)
                        print(num)
                        print(known_face)
                        is_exists = True
                        findFacePaths.append({'score' : i, 'path' : known_face})
                else:
                    known_face_encoding = []
                    num += 1
    sortedFindFacePaths = sorted(findFacePaths, key=lambda k: k['score'])   
    end_time = datetime.now()

    # Return the result as json
    result = {
        "数据库中包含：": str(num_people)+"人",
        "图片中是否包含人脸": face_found,
        "图片是否存在数据库中": is_exists,
        "数据库中相似人脸地址": sortedFindFacePaths,
        "匹配花费：": (end_time - start_time).seconds
    }
    return jsonify(result)
    #return "数据库中包含：{3}人&#10" \
    #       "图片中是否包含人脸: {0}," \
    #       "图片是否存在数据库中: {1}" \
    #       "匹配花费了：{2}".format(face_found, is_exists, (end_time - start_time).seconds, num_people)

if __name__ == "__main__":
    filePath = '/root/cs/face_recognize/photoes_serialize'
    filePaths = []
    findFacePaths = []
    load_faces(filePath, filePaths)

    app.run(host='0.0.0.0', port=5001, debug=True)



