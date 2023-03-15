from common.models import User
from django.db import models

# Create your models here.
class Content(models.Model):
    subject = models.CharField(max_length=200)
    username = models.CharField(max_length=50)
    content = models.TextField()
    content_num = models.IntegerField(default=1)
    input_img = models.ImageField(upload_to='images/', blank=True, null=True)
    content_img = models.TextField(blank=True, null=True)
    like_count = models.TextField(blank=True, null=True)
    create_date = models.DateTimeField(auto_now_add=True)
    update_date = models.DateTimeField(null=True)
    reco_img = models.TextField(blank=True, null=True)

class Answer(models.Model):
    username = models.CharField(max_length=50)
    content = models.ForeignKey(Content, on_delete=models.CASCADE)
    context = models.TextField()
    create_date = models.DateTimeField()



