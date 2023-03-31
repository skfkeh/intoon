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
pymongo_db = pymongo_connect["intoon"]
pymongo_col_folio = pymongo_db["portfolio_content"]
pymongo_col_user = pymongo_db["common_user"]


# Create your views here.
def index(request):
    try:
        answer_object = Answer.objects.filter()

        test_list = []
        for object in range(len(answer_object)):
            test_list.append(answer_object[object].content_id)
        my_dict = get_occurrence_count(test_list)
        sorted_dict = sorted(my_dict.items(), key=lambda item: item[1], reverse=True)
        sorted_dict = sorted_dict[:4]
        hot_id_list = []
        for i in range(len(sorted_dict)):
            hot_id_list.append(sorted_dict[:4][i][0])
        hot_toon_list = []
        hot_toon_img_list=[]
        for hot_id in hot_id_list:
            hot_toon = Content.objects.get(id=hot_id)
            hot_toon_img = hot_toon.content_img.split(',')[0][2:-1]
            hot_toon_list.append(hot_toon)
            hot_toon_img_list.append(hot_toon_img)
        # print('hot_toon_img_list:',hot_toon_img_list)
        hot_final_dict = zip(hot_id_list,hot_toon_list,hot_toon_img_list)
        hot_final_list = {"hot_final_dict":hot_final_dict}

        # print("hot_final_list:", hot_final_list)
        return render(request, 'base.html',hot_final_list)
    except:
        return render(request, 'base.html')

def get_occurrence_count(my_list):
  new_list = {}
  for i in my_list: 
    try: new_list[i] += 1
    except: new_list[i] = 1
  return(new_list)

def loading(request):
    return render(request, 'loading.html')

def loading3(request):
    return render(request, 'portfolio/loading3.html')

def mypage(request):
    ## 현재 로그인한 계정의 게시글만 보이도록 한정
    current_user = User.objects.get(username=request.user)
    current_content = Content.objects.filter(username=current_user)

    content_img_list = []
    content_link_list = []

    for img in range(len(current_content)):
        if len(current_content[img].content_img) <= 0:
            index = len(current_content[img].content_img)
        else:
            index = current_content[img].content_img[1:-1].find(',')
        content_first_img = current_content[img].content_img[2:index]
        content_img_list.append(content_first_img)          ## 이미지 경로값
        content_link_list.append(current_content[img].id)   ## 이미지 id값
    
    ## 최근에 작성한 글이 위에서 보이도록 reverse
    content_img_list.sort(reverse=True)
    content_link_list.sort(reverse=True)
    context = {"content_list": dict(zip(content_img_list, content_link_list)), "content_len":len(dict(zip(content_img_list, content_link_list)))}
    return render(request, 'portfolio/mypage.html', context)

def write_form(request):
    return render(request, 'portfolio/write.html')

def lounge(request):
    if request.user.is_authenticated == False:    ## 비로그인 상태에서 라운지 접속하기 위한 세팅
        lounge_content = Content.objects.filter() ## 모든 게시글을 불러온다
    else:
        current_user = User.objects.get(username=request.user)
        lounge_content = Content.objects.filter(~Q(username=current_user))

    lounge_img_list = []
    lounge_link_list = []

    for img in range(len(lounge_content)):
        index = lounge_content[img].content_img[1:-1].find(',')
        content_first_img = lounge_content[img].content_img[2:index]
        lounge_img_list.append(content_first_img)
        lounge_link_list.append(lounge_content[img].id)

    ## 최신글이 최상단에 나오도록 변경
    lounge_img_list = reversed(lounge_img_list)
    lounge_link_list = reversed(lounge_link_list)

    context = {"lounge_list": dict(zip(lounge_img_list, lounge_link_list))}

    return render(request, 'portfolio/lounge.html', context)

def content_detail(request, id):
    content = Content.objects.get(id=id)
    recommendation_result = content.reco_img
    context = {'content': content, "recommendation_result":recommendation_result}
    return render(request, 'portfolio/folio_content_detail.html', context)

def content_create(request):
    current_user = User.objects.get(username=request.user)
    current_user = str(current_user)
    current_content = Content.objects.filter(username=current_user)
    max_list = []
    iter_result = list(pymongo_col_folio.find())
    for i in iter_result:
        max_list.append(i['content_num'])
    print("max_list:", max_list)
    if request.method == 'POST':
        form = ContentForm(request.POST, request.FILES)
        if form.is_valid():
            # redirect('portfolio:loading3')
            content = form.save(commit=False)
            if len(max_list) == 0:
                content.content_num = 1
            else:
                content.content_num = int(max_list[-1]) + 1
            print("-------form---")
            content.create_date = timezone.now()
            content.save()
            input_img = str(os.getcwd()) + "/media/" + str(content.input_img)
            txt = content.content
            # content.content_for_posting =
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







