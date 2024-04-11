from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.readers.web import RssReader

import re


Settings.embed_model = resolve_embed_model("local:BAAI/bge-large-en-v1.5")
print("Loaded embedding model")

Settings.llm = Ollama(model="mistral", request_timeout=300.0, temperature=0.1)
print("Loaded primary model")

documents = RssReader(html_to_text=True).load_data([f'https://thettlpodcast.com/feed/?paged={p}' for p in range(1, 3)])
print(f"Loaded extra documents ({len(documents)}; {sum([len(d.get_text()) for d in documents])} bytes)")

for d in documents:
    if m := re.search(r'(S\d+E\d+)', d.metadata['title']):
        d.metadata['episode'] = m.group(1)

documents.append(Document(text=f"""
You have transcripts for {len(documents)} episodes of the TTL podcast
"""))

documents.append(Document(text=f"""
The name TTL Podcast is the same as the Tactics for Tech Leadership Podcast.

Mon-Chaio Lo (he/him) is an engineering executive with 20+ years of leading engineering organizations at startups (seed to series E)
and at the largest tech companies in the world (Microsoft, Uber, Meta). He has worked in a variety of domains, including
e-commerce, social media, online advertising, and healthcare. Mon-Chaio studied Computer Science and Biochemstry at the
University of Washington.

Andrew Parker (he/him) is an Engineering Leader and Manager with over 20 years of experience leading organisations in various domains,
from e-commerce to open-source configuration management software. Andy has an MSc in Software Engineering from the 
Technical University of Munich and a BSc in Computer Science from the University of Washington. His engineering
department at TIM Group was featured in two separate books on building effective teams (and had a cameo appearance in a
third).
"""))

index = VectorStoreIndex(SentenceSplitter.from_defaults().get_nodes_from_documents(documents))
print("Indexed documents")

chat_engine = index.as_chat_engine(system_prompt="""
You are a helpful, elderly librarian of the transcripts from the Tactics for Tech Leadership podcast. You will answer
questions about technical leadership, communication, culture, work patterns and any other subject that appears in
the transcripts. 

All of your statements must reference the episode title which contained the information you provide.

You must keep your answers short, too the point, and close to the transcripts. Use relevant quotes from the transcripts often.
""", chat_mode=ChatMode.CONTEXT, llm=Settings.llm)
chat_engine.streaming_chat_repl()
