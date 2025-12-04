# استخدم صورة بايثون أساسية متوافقة مع إصدارك
FROM python:3.11-slim

# تعيين متغيرات البيئة
ENV PYTHONUNBUFFERED 1
# (جديد) إضافة مجلد العمل الحالي إلى مسار Python لتمكين الاستيراد
ENV PYTHONPATH=/usr/src/app

# تعيين مجلد العمل في الحاوية
WORKDIR /usr/src/app

# نسخ ملفات المتطلبات وتثبيت الاعتماديات أولاً (للتخزين المؤقت)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# نسخ باقي ملفات المشروع إلى مجلد العمل
COPY . .

# تشغيل الخادم مع تحديد WSGI والإعدادات
# ملاحظة: المسار الصحيح هو trading_backend.config.wsgi
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "trading_backend.config.wsgi:application"]