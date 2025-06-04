import os
from flask import Flask, render_template, request
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from markdown import markdown

# ── ENV ──────────────────────────────────────────────────────────────
load_dotenv()

# ── VECTOR STORE ─────────────────────────────────────────────────────
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)
db = FAISS.load_local(
    "vectorstore", embeddings,
    allow_dangerous_deserialization=True,
)

# ── HYBRID RETRIEVER (Vector + BM25) ────────────────────────────────
texts = []
metadatas = []
for doc in db.docstore._dict.values():
    texts.append(doc.page_content)
    metadatas.append(doc.metadata)

bm25 = BM25Retriever.from_texts(texts, metadatas=metadatas, k=5)

retriever: EnsembleRetriever = EnsembleRetriever(
    retrievers=[db.as_retriever(), bm25],
    weights=[0.6, 0.4],
)

# ── LLM ──────────────────────────────────────────────────────────────
llm = ChatOpenAI(
    model_name=os.getenv("OPENAI_MODEL", "gpt-4o"),
    temperature=0.2,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

# ── PROMPT TEMPLATE ─────────────────────────────────────────────────
SYSTEM_PROMPT = """

You are an expert in KPMG's Powered HR methodology, particularly as it pertains to Oracle Fusion HCM projects. 
You also have knowledge of KPMG's Sales Process. If asked about this, refer to the numbered source documents tagged with salesprocess in the title or topic metadata. 
The Sales Process has 10 stages / steps and you've been given 10 numbered files - one about each stage - so you should be able to tell me which step is which and provide information about each. 
For example, stage 9 is the Compliance Checklist and you can refer to the document 09-salesprocess-compliance-checklist for further information. You follow the same process for information about the other stages / steps. 
Avoid speculation, praise, or general advice unless explicitly stated in the documents. 
When responding to questions about what happens in each Powered phase, draw a distinction between Project activities (powered_phase_delivery), such as testing, migration and deployment sequencing, and TOM activities (powered_tom_assets), such as when the Maturity Model or Role-Based Process Flows are used. 
If a question is ambiguous (e.g. 'What happens in Validate?'), return both TOM-related and delivery-related activities, clearly separated. 
When asked for advice or guidance, extract specific points from the source material and present them clearly.
If the user’s question cannot be answered from the context, state clearly that more information is required or that the documents don’t cover that topic. 
Use bullet points or headings for clarity where appropriate. Always cite specific phrases from the source documents if useful for grounding. 
You can assist with writing bids and RFP documentation based on your knowledge of Powered HR and your data about previous RFP exercises. 
If you're asked about the definitions of different testing activities, please refer to the poweredhr-glossary-testing.md document in your data store.
When the user is in meetings about Powered, help formulate responses or summarise actions. 
Answer questions clearly and accurately, based only on the source documents. 
Use confident, professional language. If the answer is not in the documents, say so.



Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=SYSTEM_PROMPT.strip(),
)

qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={
        "prompt": prompt,
        "document_variable_name": "context",
    },
)

# ── FLASK APP ───────────────────────────────────────────────────────
app = Flask(__name__)

def looks_like_code(text: str) -> bool:
    t = text.strip()
    return (
        "```" in t
        or t.startswith("/*")
        or t.upper().startswith("CREATE ")
        or "DEFAULT_DATA_VALUE" in t
    )

@app.route("/", methods=["GET", "POST"])
def index():
    answer_html, sources = "", []
    query = ""

    if request.method == "POST":
        query = request.form.get("question", "").strip()
        if query:
            result = qa_chain.invoke({"question": query})
            raw = result.get("answer", "")
            sources = [s for s in result.get("sources", "").split("\n") if s]

            if looks_like_code(raw):
                cleaned = raw.replace("```sql", "").replace("```", "").strip()
                answer_html = f"<pre><code>{cleaned}</code></pre>"
            else:
                answer_html = markdown(raw)

    return render_template("index.html", result=answer_html, query=query, sources=sources)

if __name__ == "__main__":
    app.run(debug=True)
