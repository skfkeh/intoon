from django.urls import path

import common.views
from . import views

app_name = 'portfolio'

urlpatterns = [
    path('', views.index, name='index'),
    path('mypage/', views.mypage, name='mypage'),
    path('loading3/', views.loading3, name='loading3'),
    path('write/', views.write_form, name='write'),
    path('loading_page/', views.loading, name='loading'),
    path('content/detail/<int:id>/', views.content_detail, name='content_detail'),
    # path('content/detail/<int:id>/', views.content_detail2, name='content_detail2'),
    path('content/create/', views.content_create, name='content_create'),
    path('lounge/', views.lounge, name='lounge'),
    path('answer/create/<int:content_id>/', views.answer_create, name='answer_create'),
    # path('like/<int:content_id>', views.likes, name='likes')
    # path('content/reco/<int:id>/', views.content_reco, name='content_reco')
    # path('answer_create/', views.answer_create, name='answer_create')
]
