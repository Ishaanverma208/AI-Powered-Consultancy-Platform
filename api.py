import requests
import os
from dotenv import load_dotenv

API_KEY = "AIzaSyAcL9pL6J2gffKSbrsVFFnQ7L56fY5tGkM"

url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent?key={API_KEY}"



payload = {
    "contents": [
        {
            "parts": [
                {"text": "Write a 2-sentence story about a robot in space."}
            ]
        }
    ]
}

headers = {"Content-Type": "application/json"}

response = requests.post(url, headers=headers, json=payload)
print(response.status_code)
print(response.text)
