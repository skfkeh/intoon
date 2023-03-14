from django.contrib import admin
from .models import Content

class ContentAdmin(admin.ModelAdmin):
    search_fields = ['subject']

class ContentAdminWriter(admin.ModelAdmin):
    search_fields = ['user_id']

class ContentAdminContent(admin.ModelAdmin):
    search_fields = ['content']

# Register your models here.
admin.site.register(Content, ContentAdmin)