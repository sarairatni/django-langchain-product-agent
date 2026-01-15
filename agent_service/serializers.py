from rest_framework import serializers
from .groq_agent import RecommendationSchema

class ProfileInputSerializer(serializers.Serializer):
    name = serializers.CharField(max_length=100)
    age = serializers.IntegerField(min_value=18)
    sector = serializers.CharField(max_length=100)
    need_description = serializers.CharField(max_length=500)

class RecommendationOutputSerializer(serializers.Serializer):
    product_id = serializers.CharField()
    justification_courte = serializers.CharField()
    score_confiance = serializers.FloatField()
    
    @classmethod
    def from_pydantic(cls, pydantic_obj: RecommendationSchema):
        return cls(pydantic_obj.model_dump())