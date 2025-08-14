import requests
import json
import subprocess

# ---------- CONFIGURATION ----------
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
GROQ_API_KEY = "YOUR_GROQ_API_KEY"
OLLAMA_MODEL = "llama3"  # Or any model you have installed in Ollama
OLLAMA_HOST = "http://localhost:11434"

# ---------- GEMINI ----------
def query_gemini(prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 200:
        raise Exception(f"Gemini Error: {response.text}")
    return response.json()["candidates"][0]["content"]["parts"][0]["text"]

# ---------- GROQ ----------
def query_groq(prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mixtral-8x7b-32768",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 200:
        raise Exception(f"Groq Error: {response.text}")
    return response.json()["choices"][0]["message"]["content"]

# ---------- OLLAMA ----------
def query_ollama(prompt):
    data = {"model": OLLAMA_MODEL, "prompt": prompt}
    response = requests.post(f"{OLLAMA_HOST}/api/generate", json=data, stream=False)
    if response.status_code != 200:
        raise Exception(f"Ollama Error: {response.text}")
    return response.json()["response"]

# ---------- MAIN RUNTIME ----------
if __name__ == "__main__":
    while True:
        prompt = input("\nEnter your question (or 'quit'): ")
        if prompt.lower() == "quit":
            break
        
        llm_choice = input("Choose LLM (gemini/groq/ollama): ").lower()

        try:
            if llm_choice == "gemini":
                answer = query_gemini(prompt)
            elif llm_choice == "groq":
                answer = query_groq(prompt)
            elif llm_choice == "ollama":
                answer = query_ollama(prompt)
            else:
                print("Invalid choice.")
                continue

            print("\n--- AI RESPONSE ---\n")
            print(answer)

        except Exception as e:
            print(f"Error: {e}")
