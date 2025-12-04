# استخدم صورة بايثون أساسية متوافقة مع إصدارك
FROM python:3.11-slim

# تعيين متغيرات البيئة
ENV PYTHONUNBUFFERED 1
ENV DJANGO_SETTINGS_MODULE=trading_backend.settings

# تعيين مجلد العمل في الحاوية
WORKDIR /usr/src/app

# نسخ ملفات المتطلبات وتثبيت الاعتماديات أولاً (للتخزين المؤقت)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# نسخ باقي ملفات المشروع إلى مجلد العمل
COPY . .

# تشغيل الخادم
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "trading_backend.wsgi"]