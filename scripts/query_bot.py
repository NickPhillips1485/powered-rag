import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain

load_dotenv()

def main():
    # Load vectorstore
    vectorstore = FAISS.load_local("vectorstore", OpenAIEmbeddings(), allow_dangerous_deserialization=True)

    # Define system prompt
    system_prompt = (
        "You are an expert in KPMG's Powered HR methodology, particularly as it pertains to Oracle Fusion HCM projects. "
        "You also have knowledge of KPMG's Sales Process. If asked about this, refer to the numbered source documents tagged with salesprocess in the title or topic metadata. "
        "The Sales Process has 10 stages / steps and you've been given 10 numbered files - one about each stage - so you should be able to tell me which step is which and provide information about each. "
        "For example, stage 9 is the Compliance Checklist and you can refer to the document 09-salesprocess-compliance-checklist for further information. You follow the same process for information about the other stages / steps. "
        "Avoid speculation, praise, or general advice unless explicitly stated in the documents. "
        "When responding to questions about what happens in each Powered phase, draw a distinction between Project activities (powered_phase_delivery), such as testing, migration and deployment sequencing, and TOM activities (powered_tom_assets), such as when the Maturity Model or Role-Based Process Flows are used. "
        "If a question is ambiguous (e.g. 'What happens in Validate?'), return both TOM-related and delivery-related activities, clearly separated. "
        "When asked for advice or guidance, extract specific points from the source material and present them clearly. "
        "If the user’s question cannot be answered from the context, state clearly that more information is required or that the documents don’t cover that topic. "
        "Use bullet points or headings for clarity where appropriate. Always cite specific phrases from the source documents if useful for grounding. "
        "You can assist with writing bids and RFP documentation based on your knowledge of Powered HR and your data about previous RFP exercises. "
        "When the user is in meetings about Powered, help formulate responses or summarise actions. "
        "Answer questions clearly and accurately, based only on the source documents. "
        "Use confident, professional language. If the answer is not in the documents, say so."
    )

    # Prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    # Initialise LLM
    llm = ChatOpenAI(model="gpt-4o")

    # Retriever – include all relevant topics
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 6,
            "filter": {
                "topic": {
                    "$in": [
                        "kpmg_sales_process",
                        "powered_phase_delivery",
                        "powered_testing_glossary",
                        "powered_tom_assets",
                        "powered_methodology_structure"
                    ]
                }
            }
        }
    )

    # Build the chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        condense_question_prompt=prompt,
        return_source_documents=True
    )

    chat_history = []

    print("🔍 Powered HR Query Assistant (type 'exit' to quit)\n")

    while True:
        query = input("🧠 Ask a question: ")
        if query.lower() in {"exit", "quit"}:
            break

        result = qa_chain({"question": query, "chat_history": chat_history})
        answer = result["answer"]
        sources = result.get("source_documents", [])
        chat_history.append((query, answer))

        print(f"\n💬 Answer: {answer.strip()}\n")

        if sources:
            print("📚 Sources:")
            for doc in sources:
                print(f" - {doc.metadata.get('source')} ({doc.metadata.get('topic', 'untagged')})")
        print("\n" + "-" * 80 + "\n")

if __name__ == "__main__":
    main()



