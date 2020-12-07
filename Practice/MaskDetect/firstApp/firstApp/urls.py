from django.contrib import admin
from django.urls import path
from django.conf.urls import url
from app import views
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('admin/', admin.site.urls),
    url('^$',views.index,name='homepage'),
    url('predictImage',views.predictImage,name='predictImage')
]

