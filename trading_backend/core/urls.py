from django.urls import path
from . import views
from .views import predict_view

urlpatterns = [
    path("predict/", predict_view, name="predict"),
    #path('register/', views.register, name='register'),
    #path('login/', views.login, name='login'),
]
from django.urls import path


