import os
import requests
import google.generativeai as genai
from groq import Groq

# ------------ OLLAMA ------------
def query_ollama(model, prompt):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()["response"]

# ------------ GEMINI ------------
def query_gemini(model, prompt):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model_obj = genai.GenerativeModel(model)
    response = model_obj.generate_content(prompt)
    return response.text

# ------------ GROQ ------------
def query_groq(model, prompt):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model
    )
    return chat_completion.choices[0].message.content

# ------------ MAIN FUNCTION ------------
def ask_llm(provider, model, prompt):
    if provider == "ollama":
        return query_ollama(model, prompt)
    elif provider == "gemini":
        return query_gemini(model, prompt)
    elif provider == "groq":
        return query_groq(model, prompt)
    else:
        raise ValueError("Invalid provider. Choose from: ollama, gemini, groq")
