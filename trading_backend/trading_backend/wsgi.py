"""
WSGI config for trading_backend project.
... (باقي التعليقات)
"""

import os
import sys # 👈 إضافة استيراد sys
import pathlib # 👈 إضافة استيراد pathlib (أكثر حداثة)

from django.core.wsgi import get_wsgi_application


# 🚨 الأسطر الجديدة: إضافة مسار جذر المشروع إلى مسار بايثون 🚨

# تحديد مسار المجلد الحالي (حيث يوجد wsgi.py)
current_path = pathlib.Path(__file__).parent.resolve()

# إضافة المجلد الرئيسي للمشروع (الذي يحتوي على مجلد trading_backend)
# هذا يعالج مشكلة الهيكل المتداخل
sys.path.append(str(current_path.parent)) 

# -----------------------------------------------------------------


os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'trading_backend.settings')

application = get_wsgi_application()