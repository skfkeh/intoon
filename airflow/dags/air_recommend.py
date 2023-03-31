from airflow.models import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.mysql.operators.mysql import MySqlOperator
from pymongo import MongoClient
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.task_group import TaskGroup

from datetime import datetime
from pandas import json_normalize
import argparse, os, re, sys
# import recommendation
sys.path.append('/home/dhj9842/venv/mysite/portfolio')
from recommendation import img_recommendation

pymongo_connect = MongoClient("mongodb://localhost:27017")
pymongo_db = pymongo_connect["intoon_fin"]
pymongo_col_folio = pymongo_db["portfolio_content"]


default_args = {
    'start_date': datetime(2023, 3, 1)
}


def reco_connect(id):
    try:
        db = pymongo_connect['intoon_fin']
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
        recommendation_result = img_recommendation.img_recommendation_func(path_list,detail_first_img)
        return recommendation_result

    except Exception as e:
        print(e, "Unable to connect to the server.")


def reco_imgpath_insert():
    total_result = list(pymongo_col_folio.find())

    for column in total_result:
        print("000000000000000000000000000000000000")
        print(f"column['id']: {column['id']}")
        if column['content_img'] is None:
            continue
        else:
            reco_list = reco_connect(str(column['id']))
            if reco_list != None:
                reco_list = reco_list

            update_query = {'id':column['id']}
            new_value = {"$set":{'reco_img':reco_list}}
            pymongo_col_folio.update_one(update_query, new_value)


with DAG(
    dag_id = 'air_recommendation', 
    schedule_interval='0 * * * *', 
    default_args=default_args, 
    catchup=False
) as dag:
    start = DummyOperator(task_id='start')
    end = DummyOperator(task_id='end')

    ### Database(MySQL)에 log 남기기
    with TaskGroup('processing_insert_db') as processing_insert_db:
        # Create Mysql Table
        creating_table = MySqlOperator(
            task_id='creating_table',
            mysql_conn_id='airflow_db',
            sql = '''
                CREATE TABLE IF NOT EXISTS recommend_success_log (
                    AIR_ID int NOT NULL AUTO_INCREMENT,
                    CONTENT VARCHAR(100),
                    CREATE_DATE datetime DEFAULT NOW(),
                    PRIMARY KEY(AIR_ID)
                );
                '''
        )

        # Insert Mysql Table
        insert_db_record = MySqlOperator(
            task_id='insert_db_record',
            mysql_conn_id='airflow_db',
            sql = '''
                INSERT INTO recommend_success_log(CONTENT)
                    VALUES ('admin');
                '''
        )

    folio_sort_func = PythonOperator(task_id='folio_sort_func', python_callable=reco_imgpath_insert, dag=dag)

    # start >> processing_insert_db >> folio_sort_func >> end
    start >> folio_sort_func >> processing_insert_db >> end