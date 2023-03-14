from common.models import User
from django.shortcuts import render, get_object_or_404, redirect
from django.utils import timezone
from django.http import HttpResponseNotAllowed
from .models import Content, Answer
from .forms import ContentForm, AnswerForm
from django.core.paginator import Paginator
from pymongo import MongoClient
from .stable_diffusion import *
from .recommendation import *
import os, re
from django.db.models import Max
from django.db.models import Q

pymongo_connect = MongoClient("mongodb://localhost:27017")
pymongo_db = pymongo_connect["project"]
pymongo_col_folio = pymongo_db["portfolio_content"]
pymongo_col_user = pymongo_db["common_user"]


# Create your views here.
def index(request):
    return render(request, 'base.html')

def loading(request):
    return render(request, 'loading.html')

def loading3(request):
    return render(request, 'portfolio/loading3.html')

def mypage(request):
    current_user = User.objects.get(username=request.user)
    current_content = Content.objects.filter(username=current_user)
    context = {"content_list": current_content}
    return render(request, 'portfolio/mypage.html', context)

def write_form(request):
    return render(request, 'portfolio/write.html')

def lounge(request):
    current_user = User.objects.get(username=request.user)
    lounge_content = Content.objects.filter(~Q(username=current_user))

    lounge_img_list = []
    lounge_link_list = []

    for img in range(len(lounge_content)):
        index = lounge_content[img].content_img[1:-1].find(',')
        content_first_img = lounge_content[img].content_img[2:index]
        lounge_img_list.append(content_first_img)
        lounge_link_list.append(lounge_content[img].content_num)

    context = {"lounge_list": dict(zip(lounge_img_list, lounge_link_list))}

    return render(request, 'portfolio/lounge_test.html', context)

def content_detail(request, id):
    content = Content.objects.get(id=id)

    other_content_img = []
    liked_count = []
    total_result = list(pymongo_col_folio.find())

    for column in total_result:
        if column['content_img'] is None:
            continue
        else:
            if id == column['id']:
                current_content_img = re.sub("'", "", column['content_img'][1:-1])
                current_content_img = current_content_img.split(', ')
            else:
                current_content_img2 = re.sub("'", "", column['content_img'][1:-1])
                current_content_img2 = current_content_img2.split(', ')
                for current_content_imgs in current_content_img2:
                    if current_content_imgs =='':
                        pass
                    else:
                        other_content_img.append(current_content_imgs)
                liked_count.append(column['like_count'])

    # print('other_content_img:',other_content_img)
    # print('len(other_content_img):', len(other_content_img))
    # print("-------column 'likes_count'-------")
    # print(liked_count)

    # other_content_img_20 = other_content_img[:20]
    # print("===================")
    # print('str(other_content_img_20):',str(other_content_img_20))
    # print('str(other_content_img_20):',replace(str(other_content_img_20)), '"')
    # print(len(other_content_img_20))
    # print(count(other_content_img_20))

    total_result = list(pymongo_col_folio.find())
    # 현재 게시물 이미지 경
    detail_first_img = current_content_img[0]
    # 20개 이미지 경로
    path_list = other_content_img[-20:-1]
    print(f"path_list:{path_list}")
    recommendation_result = img_recommendation.img_recommendation_func(path_list,detail_first_img)
    reco_link_list = []
    print(recommendation_result)
    print(type(recommendation_result))
    print(recommendation_result[0].content_num)
    for img in range(len(recommendation_result)):
        reco_link_list.append(recommendation_result[img].content_num)

    context = {'content': content, 'recommendation_result': recommendation_result, "reco_list": dict(zip(recommendation_result, reco_link_list))}
    return render(request, 'portfolio/folio_content_detail.html', context) #, detail_first_img, path_list)


def content_create(request):
    current_user = User.objects.get(username=request.user)
    current_user = str(current_user)
    current_content = Content.objects.filter(username=current_user)
    max_list = []
    iter_result = list(pymongo_col_folio.find())
    for i in iter_result:
        max_list.append(i['content_num'])
    if request.method == 'POST':
        form = ContentForm(request.POST, request.FILES)
        if form.is_valid():
            # redirect('portfolio:loading3')
            content = form.save(commit=False)
            content.content_num = int(max_list[-1]) + 1
            print("-------form---")
            content.create_date = timezone.now()
            content.save()
            input_img = str(os.getcwd()) + "/media/" + str(content.input_img)
            txt = content.content

            if input_img == "/home/dhj9842/venv/mysite/media/":
                new_path_list = Txt2img.txt2img_func(txt, current_user)
            else:
                new_path_list = Img2img.img2img_func(input_img, txt, current_user)

            # print(f"new_path_list: {new_path_list}")
            # print(f"type new_path_list: {type(new_path_list)}")

            content.content_img = str(new_path_list)

            ################# content.reco_img = str(new_path_list)################


            content.save()

            return redirect('portfolio:mypage')
    else:
        form = ContentForm()
    context = {'form': form}
    return render(request, '/portflio/mypage.html', context)

def answer_create(request, content_id):
    current_user = User.objects.get(username=request.user)
    current_user = str(current_user)
    content = get_object_or_404(Content, pk=content_id)
    if request.method == "POST":
        form = AnswerForm(request.POST)
        if form.is_valid():
            answer = form.save(commit=False)
            answer.username = current_user
            answer.content = content
            answer.create_date = timezone.now()
            answer.context = request.POST.get('context')
            answer.save()
            return redirect('portfolio:content_detail', id=content.id)
    else:
        return HttpResponseNotAllowed('Only POST is possible.')
    context = {'content': content, 'form': form}
    return render(request, 'portfolio:folio_content_detail', context)


# def likes(request, content_id):
#     try:
#         if request.user.is_authenticated:
#             print('00000000000000000000000')
#             content = get_object_or_404(Content, pk=content_id)
#             print('11111111111111111')
#             print(content.like_users)
#             if content.like_users.filter(pk=request.user.username).exists():
#                 print('333333333333333333')
#                 content.like_users.remove(request.user)
#             else:
#                 print('444444444444444')
#                 content.like_users.add(request.user)
#             return redirect('/portflio/folio_content_detail.html')
#     except:
#         return redirect('common:login')
#



# def content_reco(request, id):
#     print('aaaaaa')
#     model_url = './rec_tf_model'
#
#     total_result = list(pymongo_col_folio.find())
#     other_content_img = []
#
#     for column in total_result:
#         if column['content_img'] is None:
#             continue
#         else:
#             if id != column['id']:
#                 other_content_img.append(re.sub("'", "", column['content_img'][1:-1]))
#
#     path_list = other_content_img[:20]
#     print('==================')
#     print(path_list)
#     recommendation_result = img_recommendation.img_recommendation_func(model_url,path_list)
