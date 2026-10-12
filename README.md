# Placement and Interview Study Helper (RAG Chatbot)

Upload your own notes (PDFs) and ask questions. The app finds the most relevant pages and asks an LLM to answer from them, showing the source file and page.

## How it works
1. **Read** each PDF page.
2. **Chunk** page text into 800-character pieces with 150 characters of overlap.
3. **Embed** each chunk with `all-MiniLM-L6-v2` (free, runs locally) and store it in **ChromaDB**.
4. **Retrieve** the 4 chunks closest in meaning to the question (cosine similarity).
5. **Generate** an answer with Google Gemini, using only those chunks, and show the sources.

## Setup
1. Install Python 3.10 or newer.
2. In this folder run:
   ```
   python -m venv venv
   venv\Scripts\activate          (Windows)   or   source venv/bin/activate   (Mac/Linux)
   pip install -r requirements.txt
   ```
3. Get a free API key at https://aistudio.google.com/apikey
4. Copy `.env.example` to `.env` and paste your key.
5. Run:
   ```
   streamlit run app.py
   ```
6. Upload a PDF in the sidebar, click **Build index**, then ask a question.

The first run downloads the embedding model (about 90 MB) and takes a minute.

## Measuring accuracy
Edit `questions.csv` with 15-20 questions and a keyword that a correct answer must contain, then run `python eval.py`. The score is your real evaluation number.

## Tech stack
Python, Streamlit, ChromaDB (with its built-in all-MiniLM-L6-v2 embeddings), pypdf, Google Gemini API.

## Results
- Documents indexed: [fill in]
- Test questions: [fill in]
- Correct answers: [fill in]%

## Demo
[Add screenshots and your live link here]
