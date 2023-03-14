# Generated by Django 4.1.7 on 2023-03-14 05:14

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('portfolio', '0003_remove_content_like_counts_content_like_count'),
    ]

    operations = [
        migrations.AddField(
            model_name='content',
            name='like_users',
            field=models.ManyToManyField(related_name='like_articles', to=settings.AUTH_USER_MODEL),
        ),
        migrations.AlterField(
            model_name='content',
            name='username',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL),
        ),
    ]