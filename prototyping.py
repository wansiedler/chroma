import chromadb


client = chromadb.PersistentClient(path="test.db")

collection = client.get_or_create_collection("test")

collection.add(
    ids=["1", "2", "3"],
    documents=["Hello", "World", "Hello World"],
    metadatas=[{"source": "test"}, {"source": "test"}, {"source": "test"}],
    # embeddings=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
)
