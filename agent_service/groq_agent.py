from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_groq import ChatGroq 
import os
from dotenv import load_dotenv

load_dotenv() 
GROQ_API_KEY = os.getenv("GROQ_API_KEY") 

if not GROQ_API_KEY:
    raise ValueError("La clé GROQ_API_KEY n'est pas configurée dans l'environnement.")

class RecommendationSchema(BaseModel):
    product_id: str = Field(description="L'identifiant unique du produit/service recommandé (ex: S-2024-PRO).")
    justification_courte: str = Field(description="Explication concise (max 30 mots) du pourquoi ce produit est idéal pour le client.")
    score_confiance: float = Field(description="Un score de confiance entre 0.0 et 1.0 sur la pertinence de la recommandation.")

parser = PydanticOutputParser(pydantic_object=RecommendationSchema)

llm = ChatGroq(
    model="llama-3.3-70b-versatile", 
    api_key=GROQ_API_KEY, 
    temperature=0.1
)
 
def generate_recommendation_chain():
    template = """
Tu es un Expert en Recommandation de Produits pour des clients B2B.
Ton rôle est d'analyser le profil client fourni et de recommander un seul produit/service.

RÈGLES DE CONFIANCE (CRITIQUE pour le score_confiance) :
1. CONFIANCE ÉLEVÉE (Score > 0.80) : Si le secteur et le besoin correspondent parfaitement à l'un des produits.
2. CONFIANCE MOYENNE (Score entre 0.70 et 0.80) : Si le besoin est clair mais le secteur est générique.
3. CONFIANCE BASSE (Score < 0.70) : Si le besoin est ambigu, flou ou incohérent.

Tu dois absolument respecter le format de sortie JSON spécifié par les instructions de formatage.
Ne rajoute aucun texte avant ou après le JSON.

Profil Client :
- Âge du client : {age}
- Secteur d'activité : {sector}
- Besoin exprimé : {need}

Liste des Produits/Services disponibles :
- S-2024-PRO (Solution Pro Data) : Idéale pour les besoins complexes en analyse de données.
- B-2024-ESS (Basique Essentiel) : Pour les petites entreprises ayant des besoins simples en gestion.
- C-2024-MIG (Migration Cloud) : Pour les clients cherchant à moderniser leur infrastructure.

{format_instructions}
"""
    prompt = ChatPromptTemplate.from_template(
        template=template,
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    chain = prompt | llm | parser
    return chain

def recommend_product(age: int, sector: str, need: str) -> RecommendationSchema:
    try:
        chain = generate_recommendation_chain()
        result = chain.invoke({
            "age": age,
            "sector": sector,
            "need": need
        })
        return result
    except Exception as e:
        print(f"Erreur lors de l'appel LLM: {e}")
        return RecommendationSchema(
            product_id="API_FAIL",
            justification_courte=f"Erreur API Groq: {str(e)}",
            score_confiance=0.0
        )

if __name__ == '__main__':
    test_age = 45
    test_sector = "Finance"
    test_need = "J'ai besoin d'une solution pour analyser rapidement les gros volumes de données de marché."
    
    print(f"Analyse du profil : {test_sector}, {test_age}, Besoin: '{test_need[:50]}...'")
    
    recommendation = recommend_product(test_age, test_sector, test_need)
    
    print("\n--- RÉSULTAT DE L'AGENT (GROQ) ---")
    print(f"Produit Recommandé : {recommendation.product_id}")
    print(f"Justification : {recommendation.justification_courte}")
    print(f"Confiance : {recommendation.score_confiance:.2f}")
    print(f"Type de l'objet retourné : {type(recommendation)}")
    print("\n--- TEST TERMINE ---")