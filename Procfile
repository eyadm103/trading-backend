web: gunicorn trading_backend.wsgi --env DJANGO_SETTINGS_MODULE=trading_backend.settings --log-file -
worker: python live_trade_runner.py