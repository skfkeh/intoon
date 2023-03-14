from django import forms
from portfolio.models import Content, Answer

class ContentForm(forms.ModelForm):
    class Meta:
        model = Content
        fields = ['subject', 'username', 'content', 'input_img', 'content_img', 'like_count']
        labels = {
            'subject': '제목',
            'username': '작성자',
            'content': '내용',
            'input_img': '입력이미지',
            'content_img': '이미지',
        }

class AnswerForm(forms.ModelForm):
    class Meta:
        model = Answer
        fields = ['context']
        labels = {
            'context': '답변내용',
        }
