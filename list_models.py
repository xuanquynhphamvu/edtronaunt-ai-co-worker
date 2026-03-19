import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

print("=== DANH SÁCH MODEL CÓ SẴN CHO BẠN ===\n")

for model in genai.list_models():
    print(model.name)
    if 'generateContent' in model.supported_generation_methods:
        print("   → Hỗ trợ chat (có thể dùng được!)")
    print("-" * 50)