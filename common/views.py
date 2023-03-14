from django.contrib.auth import authenticate, login
from django.shortcuts import render, redirect
from common.forms import UserForm
from .models import User

# Create your views here.
def signup(request):
    if request.method == "POST":
        form = UserForm(request.POST)
        if form.is_valid():
            form.save()

            ### 회원가입 시, 자동 로그인
            # username = form.cleaned_data.get('username')
            # raw_password = form.cleaned_data.get('password1')
            # user = authenticate(username=username, password=raw_password)  # 사용자 인증
            # login(request, user, backend='django.contrib.auth.backends.ModelBackend') # 로그인
            return redirect('index')
    else: # Get 요청
        form = UserForm
    return render(request, 'common/signup.html', {'form':form})