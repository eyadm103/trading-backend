FROM python:3.10-slim

ENV PYTHONUNBUFFERED 1

WORKDIR /usr/src/app/trading_backend

COPY requirements.txt /usr/src/app/
RUN pip install --no-cache-dir -r /usr/src/app/requirements.txt

COPY . /usr/src/app/