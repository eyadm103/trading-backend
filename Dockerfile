FROM python:3.10-slim

ENV PYTHONUNBUFFERED 1

# *** هذا السطر هو الحل الأخير: إخبار Python بمكان البحث ***
ENV PYTHONPATH=/usr/src/app
# ******************************************************

WORKDIR /usr/src/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .