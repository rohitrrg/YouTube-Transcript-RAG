import re
from llm import Mistral
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from youtube_transcript_api.proxies import WebshareProxyConfig
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser



class YouTubeRetriever:
   
    def __init__(self):
        self.llm = Mistral()
        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        self.ytt_api = YouTubeTranscriptApi(
            proxy_config = WebshareProxyConfig(
                proxy_username="zgqpqwdi",
                proxy_password="l7uq2orsu8xi",
            )
        )
        self.prompt = PromptTemplate(
                        template="""
                        You are a helpful assistant.
                        Answer ONLY from the provided transcript context.
                        If the context is insufficient, just say you don't know.

                        {context}
                        Question: {question}
                        """,
                        input_variables = ['context', 'question']
                    )
        self.parser = StrOutputParser()
    

    def extract_video_id(self, url: str):
        if re.fullmatch(r"[A-Za-z0-9_\-]{11}", url):
            return url
        patterns = [
            r"v=([A-Za-z0-9_\-]{11})",
            r"youtu\.be/([A-Za-z0-9_\-]{11})",
            r"youtube\.com/shorts/([A-Za-z0-9_\-]{11})"
        ]
        for p in patterns:
            m = re.search(p, url)
            if m:
                return m.group(1)
        raise ValueError("Could not extract a YouTube video ID from input.")


    def load_transcript_as_docs(self, video_id: str):
        try:
            transcript_list = self.ytt_api.fetch(video_id)
            transcript = " ".join(chunk.text for chunk in transcript_list)
            return transcript
        except TranscriptsDisabled:
            raise RuntimeError("No transcript available for this video (captions may be disabled).")
    

    def split_documents(self, docs, chunk_size: int = 1200, chunk_overlap: int = 200):
        splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
        return splitter.create_documents([docs])


    def vector_store_as_retriver(self, chunks):
        vector_store = FAISS.from_documents(chunks, self.embeddings)
        return vector_store.as_retriever(search_type="similarity", search_kwargs={"k":4})


    def format_docs(self, retrieved_docs):
        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
        return context_text


    def chain(self, retriever):
       
       parallel_chain = RunnableParallel({
           'context': retriever | RunnableLambda(self.format_docs) ,
           'question': RunnablePassthrough()
           })
       main_chain = parallel_chain | self.prompt | self.llm | self.parser

       return main_chain