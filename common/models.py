from django.db import models
from django.contrib.auth.models import AbstractUser

# Create your models here.
class User(AbstractUser):
    username = models.CharField(max_length=20, primary_key=True)
    profile_img_src = models.CharField(max_length=150, blank=True)
    profile_msg = models.TextField(max_length=300, blank=True)
    # birth_date = models.DateField(null=True, blank=True)


    def __str__(self):
        return self.username