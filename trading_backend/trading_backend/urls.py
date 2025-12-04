from django.contrib import admin
from django.urls import path, include  # لاحظ أننا أضفنا include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('core.urls')),  # ده بيربط أي حاجة تبدأ بـ /api/ بالـ core.urls
]

