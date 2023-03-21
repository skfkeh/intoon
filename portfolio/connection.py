import pymongo
import re, sys
import recommendation
# sys.path.append('/home/dhj9842/venv/mysite/portfolio')
# from . import img_recommendation

# Replace the uri string with your MongoDB deployment's connection string.
conn_str = "mongodb://admin:1234@localhost:27017/"
# set a 5-second connection timeout
client = pymongo.MongoClient(conn_str, serverSelectionTimeoutMS=5000)


def connect(id):
    try:
        db = client['intoon']
        current_content_img = []
        other_content_img = []
        recommendation_result = []
        for data in db['portfolio_content'].find():
            if data['content_img'] is None:
                continue
            else:
                if id == str(data['id']):
                    current_content_img_tmp = re.sub("'", "", data['content_img'][1:-1])
                    current_content_img = current_content_img_tmp.split(', ')
                
                else:
                    current_content_img2 = re.sub("'", "", data['content_img'][1:-1])
                    current_content_img2 = current_content_img2.split(', ')
                    for current_content_imgs in current_content_img2:
                        if current_content_imgs =='':
                            pass
                        else:
                            other_content_img.append(current_content_imgs)
        # 현재 게시물 이미지 경로
        detail_first_img = current_content_img[0]

        # 20개 이미지 경로
        path_list = other_content_img[-20:-1]
        recommendation_result = recommendation.img_recommendation.img_recommendation_func(path_list,detail_first_img)
        print(recommendation_result)
        return recommendation_result

    except Exception as e:
        print(e, "Unable to connect to the server.")


if __name__=="__main__":
     connect(sys.argv[1])
