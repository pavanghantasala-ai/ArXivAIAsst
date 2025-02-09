from flask import Flask, render_template, request, jsonify
import arxiv
import pytz
from datetime import datetime, timedelta
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

app = Flask(__name__)
llm = Ollama(model="mistral")

summary_prompt = PromptTemplate(
    input_variables=["title", "abstract"],
    template="""
    Summarize this AI paper in 3 brief points:
    Title: {title}
    Abstract: {abstract}
    """
)
summary_chain = LLMChain(llm=llm, prompt=summary_prompt)

def get_recent_papers(days_back=1):
    categories = ['cs.AI', 'cs.LG', 'cs.CL']
    date_since = (datetime.now(pytz.utc) - timedelta(days=days_back))
    search_query = ' OR '.join(f'cat:{cat}' for cat in categories)
    client = arxiv.Client()
    search = arxiv.Search(query=search_query, max_results=5, sort_by=arxiv.SortCriterion.SubmittedDate)
    papers = []
    for result in client.results(search):
        if result.published.replace(tzinfo=pytz.utc) > date_since:
            summary = summary_chain.run(title=result.title, abstract=result.summary)
            papers.append({
                'title': result.title,
                'abstract': result.summary,
                'summary': summary,
                'pdf_url': result.pdf_url
            })
    return papers

@app.route('/')
def index():
    papers = get_recent_papers()
    return render_template('index.html', papers=papers)

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        user_input = request.json.get('message', '')
        response = llm.invoke(user_input)
        return jsonify({"response": response})
    return render_template('chat.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

