🚀 AI Consultancy Platform
A full-stack AI-powered consultancy platform where clients:

Enter their company name.

System fetches peer/competitor data from the internet.

Asks follow-up questions to refine requirements.

Generates structured consulting advice using Ollama, Google Gemini, and Groq APIs for speed, reasoning power, and large context handling.

✨ Features
🔍 Competitor Data Fetching – Automated internet search to retrieve latest market insights.

💬 Conversational Follow-up – Multi-step questioning to refine consulting needs.

📊 Structured Recommendations – Consulting-style actionable reports.

🧠 Multi-Model AI Power

Google Gemini 1.5 Flash → Long-context reasoning & strategic recommendations.

Groq → Super-fast Q&A & clarifications.

Ollama → Local/private AI reasoning with no API cost.

🌐 Full-Stack Implementation – Backend API with Express + Frontend in React.

🛠️ Tech Stack
Frontend → React.js (Vite)

Backend → Node.js (Express.js)

APIs → Google Gemini, Groq, Ollama

Others → Axios, Dotenv, CORS

📂 Project Structure
ai-consultancy-platform/
│
├── backend/
│   ├── server.js               # Express backend API  
│   ├── routes/
│   │   ├── ollamaRoute.js       # Handles Ollama AI calls  
│   │   ├── geminiRoute.js       # Handles Google Gemini API calls  
│   │   ├── groqRoute.js         # Handles Groq API calls  
│   ├── services/
│   │   ├── ollamaService.js  
│   │   ├── geminiService.js  
│   │   ├── groqService.js  
│   ├── .env                     # API Keys here (Do NOT commit)  
│   └── package.json  
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Main frontend logic  
│   │   ├── api.js               # Axios calls to backend  
│   │   └── styles.css  
│   ├── vite.config.js  
│   └── package.json  
│
├── README.md  
└── .gitignore  
🔑 Environment Variables
Create a .env file inside backend/ with:

GEMINI_API_KEY=your_google_gemini_api_key
GROQ_API_KEY=your_groq_api_key
OLLAMA_HOST=http://localhost:11434
⚡ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/yourusername/ai-consultancy-platform.git
cd ai-consultancy-platform
2️⃣ Backend Setup
cd backend
npm install
node server.js
3️⃣ Frontend Setup
cd frontend
npm install
npm run dev
🚀 Usage
Run the backend & frontend simultaneously.

Open the app in browser.

Enter company name.

Answer follow-up questions.

Get AI-powered consulting report instantly.

📌 API Notes
Google Gemini 1.5 Flash – For strategic consulting & long responses.

Groq – For fast Q&A & follow-ups.

Ollama – For offline/local private analysis.

📄 License
This project is licensed under the MIT License.
