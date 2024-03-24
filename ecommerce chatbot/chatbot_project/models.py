from cgitb import text
from dataclasses import field
from pydoc_data.topics import topics
from tabnanny import verbose
from django.db import models

class Response(models.Model):
    text = models.TextField()
    class meta:
        verbose_name_plural = "Responses"


class UserInput(models.Model):
    username = models.CharField(max_length=500)
    chats = models.CharField(max_length=500)

class ClientInfo(models.Model):
    username = models.CharField(max_length=500)
    item = models.CharField(max_length=500)
    paid = models.IntegerField()
    date = models.DateField()
    review = models.TextField()

class UploadImag(models.Model):
    username = models.CharField(max_length=500)
    img = models.ImageField(blank=True)
