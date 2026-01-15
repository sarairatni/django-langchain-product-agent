from django.urls import path
from .views import AgentAnalyzeAPIView, recommendation_form

urlpatterns = [
    path('analyze/', AgentAnalyzeAPIView.as_view(), name='agent-analyze'),
    path('', recommendation_form, name='recommendation-form'),
]