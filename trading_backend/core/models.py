from django.contrib.auth.models import User
from django.db import models

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    alpaca_api_key = models.CharField(max_length=100, blank=True, null=True)
    alpaca_secret_key = models.CharField(max_length=100, blank=True, null=True)
    alpaca_token = models.CharField(max_length=255, blank=True, null=True)

    def __str__(self):
        return self.user.username
