from datasets import load_dataset
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain.docstore.document import Document
from transformers import pipeline

print("Loading SQuAD dataset...")
data = load_dataset("squad", split="validation[:100]")

# Build context docs
docs = [Document(page_content=item["context"]) for item in data]
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(docs, embeddings)

pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=200)
from langchain_community.llms import HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=pipe)
qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

em, f1_total, count = 0, 0, 0

for item in data:
    question = item["question"]
    gold_answers = item["answers"]["text"]
    if not gold_answers:
        continue
    pred = qa.run(question).strip().lower()
    gold = gold_answers[0].strip().lower()

    if pred == gold:
        em += 1

    pred_tokens = set(pred.split())
    gold_tokens = set(gold.split())
    common = pred_tokens & gold_tokens
    if common:
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    f1_total += f1
    count += 1

print(f"\n✅ Results on SQuAD 2.0 (100 samples)")
print(f"Exact Match (EM): {em}%")
print(f"F1 Score: {f1_total/count:.2f}")
