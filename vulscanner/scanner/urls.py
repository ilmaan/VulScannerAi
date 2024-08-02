from django.urls import path

from . import views


urlpatterns = [
    path('',views.home,name='home'),
    path('file_scanner',views.file_scanner.as_view(),name='file_scanner'),
    path('code_scanner',views.code_scanner,name='code_scanner')

]