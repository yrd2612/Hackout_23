# Generated by Django 3.2.16 on 2023-05-13 18:10

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('signup', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='signup',
            name='password',
            field=models.CharField(default=2, max_length=50),
            preserve_default=False,
        ),
    ]
