from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.ollama import Ollama
from llama_index.readers.web import RssReader

#documents = SimpleDirectoryReader("data").load_data()
documents = RssReader(html_to_text=True).load_data([f'https://thettlpodcast.com/feed/?paged={p}' for p in range(1, 10)])
print(f"Loaded extra documents ({len(documents)})")

# bge embedding model
Settings.embed_model = resolve_embed_model("local:BAAI/bge-large-en-v1.5")
print("Loaded embedding model")

# ollama
Settings.llm = Ollama(model="mistral", request_timeout=300.0)
print("Loaded primary model")

index = VectorStoreIndex.from_documents(
    documents,
)
print("Indexed documents")

query_engine = index.as_query_engine()
while(True):
    query = input('> ')
    response = query_engine.query(query)
    print(response)
