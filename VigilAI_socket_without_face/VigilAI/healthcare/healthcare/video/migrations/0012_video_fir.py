# Generated by Django 4.2.4 on 2023-08-26 03:17

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('video', '0011_video_photos'),
    ]

    operations = [
        migrations.AddField(
            model_name='video',
            name='fir',
            field=models.CharField(default='uu', max_length=50),
        ),
    ]