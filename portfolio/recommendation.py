import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np
import os
import cv2
import time
from PIL import Image
from scipy.spatial import distance

class img_recommendation:
    # model_url = 'https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2'
    # IMAGE_SHAPE = (224, 224)
    # layer = hub.KerasLayer(model_url)
    # model = tf.keras.Sequential([layer])
    def img_recommendation_func(path_list, detail_first_img):
        print(f'path_list: {path_list}')
        path_list.insert(0,detail_first_img)
        model_url = 'rec_tf_model'
        def extract(file):
            file = '/home/dhj9842/venv/mysite' + file
            file = Image.open(file).convert('L').resize(IMAGE_SHAPE)
            file = np.stack((file,) * 3, axis=-1)
            file = np.array(file) / 255.0
            embedding = model.predict(file[np.newaxis, ...])
            vgg16_feature_np = np.array(embedding)
            flattended_feature = vgg16_feature_np.flatten()
            return flattended_feature

        def change(path_list):

            print('path_list:',path_list)
            extract_png = []
            extract_name=[]
            for png in path_list:
                print(f"png1:{png}")
                # png = '/home/dhj9842/venv/mysite' + png
                extract_png.append(extract(png))
                extract_name.append(png)
            re_file = {x:y for x,y in zip(extract_name, extract_png)}
            return re_file


        def cdist():
            cdist = []
            final_list = []
            test_names = list(test.keys())
            for value in range(len(test)):
                b = distance.cdist([list(test.values())[1]], [list(test.values())[value]], 'cosine')[0][0]
                cdist.append(b)
            sort_cdist = sorted(cdist)
            cdists = sort_cdist[1:5]
            cd_index = []
            for i in cdists:
                idx = cdist.index(i)
                cd_index.append(idx)
            for i in cd_index:
                final_list.append(test_names[i])
            return final_list


        IMAGE_SHAPE = (224, 224)
        layer = hub.KerasLayer(model_url)
        model = tf.keras.Sequential([layer])

        test = change(path_list)

        recommend_path = cdist()
        return recommend_path