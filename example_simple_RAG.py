import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer


class SimpleRAGmodel:
    def __init__(
        self, path_model_name, path_data_file_name, text_column, threshold=0.6
    ):
        self.retriever_model = self.load_model(path_model_name)
        self.documents = self.load_data(path_data_file_name, text_column)
        self.top_k = len(self.documents)
        self.threshold = threshold
        self.doc_embeddings = self.make_docs_embeddings()
        self.indexer = self.create_document_indexer()

    def load_model(self, path_model_name):
        return SentenceTransformer(path_model_name)

    def load_data(self, path_data_file_name, text_column):
        imdb_data = pd.read_csv(path_data_file_name, delimiter="\t", header=None)
        imdb_data.rename(columns={0: "reviews", 1: "score"}, inplace=True)
        return imdb_data[text_column]

    def make_docs_embeddings(self):
        return self.retriever_model.encode(
            self.documents, normalize_embeddings=True
        ).astype("float32")

    def create_document_indexer(self):
        indexer = faiss.IndexFlatIP(self.doc_embeddings.shape[1])
        indexer.add(self.doc_embeddings)
        return indexer

    def retrieve_information(self, question):
        question_embedding = self.retriever_model.encode(
            [question], normalize_embeddings=True
        ).astype("float32")
        scores, idxs = self.indexer.search(question_embedding, self.top_k)
        return [
            self.documents[i]
            for j, i in enumerate(idxs[0])
            if scores[0][j] >= self.threshold
        ]

    def activateRAG(self):
        while True:
            question = input(
                "What is your question?. Type 'exit'to close the RAG model \n"
            )
            if question.lower() != "exit":
                results = self.retrieve_information(question)
                print(f"Question: {question}\n")
                print("Top matching reviews")
                for i, text in enumerate(results, 1):
                    print(f"{i}. {text}")
            else:
                break


if __name__ == "__main__":
    rag_model = SimpleRAGmodel(
        "BAAI/bge-small-en-v1.5", "assets/imdb_labelled.txt", "reviews", 0.6
    )
    execute_rag_model = rag_model.activateRAG()
