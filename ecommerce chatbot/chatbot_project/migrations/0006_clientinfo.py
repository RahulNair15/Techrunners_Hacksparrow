# Generated by Django 4.2.11 on 2024-03-23 13:42

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('chatbot_project', '0005_userinput'),
    ]

    operations = [
        migrations.CreateModel(
            name='ClientInfo',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('username', models.CharField(max_length=500)),
                ('item', models.CharField(max_length=500)),
                ('paid', models.IntegerField()),
                ('date', models.DateField()),
                ('review', models.TextField()),
            ],
        ),
    ]
