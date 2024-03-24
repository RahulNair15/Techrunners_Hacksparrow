from django.contrib import admin
from .models import UserInput, ClientInfo, UploadImag
from chatbot_project.models import Response

# Register your models here.
admin.site.register(Response)
# admin.site.register(Topic)
@admin.register(UserInput)
@admin.register(ClientInfo)
@admin.register(UploadImag)

class ClientUserAdmin(admin.ModelAdmin):
    list_display2 = ['id', 'username', 'item', 'paid', 'date']

class UserInputAdmin(admin.ModelAdmin):
    list_display = ['id', 'username', 'chats']

class UploadImagAdmin(admin.ModelAdmin):
    list_display = ['id', 'username', 'img']