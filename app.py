import os
from datetime import datetime, timedelta
import arxiv
import pickle
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

app = Flask(__name__)

# Initialize Ollama LLM
llm = OllamaLLM(model="mistral")

# Create cache directory
cache_dir = Path('paper_cache')
cache_dir.mkdir(exist_ok=True)

# Define Prompt Templates
summary_prompt = PromptTemplate(
    input_variables=["title", "abstract"],
    template="""
    Summarize this AI paper in 3 brief points:
    Title: {title}
    Abstract: {abstract}
    Focus on practical implications and key innovations.
    """
)

qa_prompt = PromptTemplate(
    input_variables=["question", "papers_context"],
    template="""
    Based on these recent AI papers:
    {papers_context}
    
    Answer this question concisely: {question}
    Include 1-2 relevant paper references if applicable.
    """
)

summary_chain = summary_prompt | llm  # New syntax replaces LLMChain
qa_chain = qa_prompt | llm  # New syntax replaces LLMChain

def get_recent_papers(days_back=1):
    """Fetch recent AI papers from arXiv"""
    categories = ['cs.AI', 'cs.LG', 'cs.CL']
    date_since = datetime.now().astimezone() - timedelta(days=days_back)

    search_query = ' OR '.join(f'cat:{cat}' for cat in categories)
    search_query += ' AND (language model OR LLM OR artificial intelligence)'

    client = arxiv.Client()
    search = arxiv.Search(
        query=search_query,
        max_results=5,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    papers = []
    for result in client.results(search):
        paper_date = result.published.astimezone()
        if paper_date > date_since:
            papers.append({
                'id': result.entry_id,
                'title': result.title,
                'abstract': result.summary,
                'authors': [author.name for author in result.authors],
                'published': paper_date.strftime('%Y-%m-%d'),
                'pdf_url': result.pdf_url
            })
    
    return papers

def format_papers_for_ui(papers):
    """Generate summaries for UI"""
    formatted_papers = []
    
    for paper in papers:
        summary = summary_chain.invoke({
            "title": paper['title'],
            "abstract": paper['abstract']
        })
        formatted_papers.append({
            "title": paper['title'],
            "summary": summary,
            "pdf_url": paper['pdf_url']
        })

    return formatted_papers

@app.route('/')
def index():
    """Home page showing recent papers"""
    papers = get_recent_papers()
    formatted_papers = format_papers_for_ui(papers)
    return render_template('index.html', papers=formatted_papers)

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    """Chat page where users can ask questions"""
    if request.method == 'POST':
        question = request.json.get('question')

        # Load cached papers
        cache_file = cache_dir / 'papers_cache.pkl'
        if not cache_file.exists():
            return jsonify({"answer": "No recent papers available. Try again later!"})

        with open(cache_file, 'rb') as f:
            papers = pickle.load(f)

        # Prepare context from recent papers
        papers_context = "\n\n".join([
            f"Title: {paper['title']}\nAbstract: {paper['abstract']}"
            for paper in list(papers.values())[-5:]
        ])

        # Generate answer
        answer = qa_chain.invoke({
            "question": question,
            "papers_context": papers_context
        })
        return jsonify({"answer": answer})

    return render_template('chat.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
