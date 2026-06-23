import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA

load_dotenv()

CHROMA_PATH = "./chroma_db"

def build_chain():
    embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embed_model)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    return qa_chain


def chat_loop():
    print("KAIZEN RAG Task - Ask anything about the docs (type 'exit' to quit)\n")
    qa = build_chain()

    while True:
        query = input("Enter your question: ").strip()
        if query.lower() in ("exit", "quit", "q"):
            break
        if not query:
            continue

        res = qa.invoke({"query": query})

        print(f"\nBot: {res['result']}")

        shown = set()
        for doc in res["source_documents"]:
            src = doc.metadata.get("source", "?")
            if src not in shown:
                print(f"  [source: {src}]")
                shown.add(src)
        print()

if __name__ == "__main__":
    chat_loop()
