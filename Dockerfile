FROM python:3.10-slim

WORKDIR /usr/src/app/trading_backend

COPY requirements.txt /usr/src/app/
RUN pip install --no-cache-dir -r /usr/src/app/requirements.txt

COPY . /usr/src/app/

CMD ["gunicorn", "config.wsgi:application", "--bind", "0.0.0.0:${PORT}"]
