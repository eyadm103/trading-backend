from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .ai_model import get_decision

@csrf_exempt
def predict_view(request):
    if request.method == "POST":
        try:
            body = json.loads(request.body)
            features = body.get("features", [])
            decision = get_decision(features)
            return JsonResponse({"decision": decision})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)
    return JsonResponse({"error": "Only POST allowed"}, status=405)
