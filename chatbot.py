import os
# import pathlib
#
# # Get the directory of the current script
# current_directory = pathlib.Path(__file__).parent.resolve()
#
# # Construct the full path to the cacert.pem file
# cacert_path = current_directory / 'cacert.pem'
#
# # Set the environment variables
# os.environ['REQUESTS_CA_BUNDLE'] = str(cacert_path)
# os.environ['SSL_CERT_FILE'] = str(cacert_path)

os.environ["OPENAI_API_KEY"] = "Add your key here"
os.environ["PINECONE_API_KEY"] = "Add your key here"

import pinecone_datasets

dataset = pinecone_datasets.load_dataset('wikipedia-simple-text-embedding-ada-002-100K')

# drop metadata column and renamed blob to metadata
dataset.documents.drop(['metadata'], axis=1, inplace=True)
dataset.documents.rename(columns={'blob': 'metadata'}, inplace=True)

# sample 30k documents
dataset.documents.drop(dataset.documents.index[50_000:], inplace=True)

from pinecone import Pinecone, ServerlessSpec

# configure client
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# configure serverless spec
spec = ServerlessSpec(cloud='aws', region='us-east-1')

# check for and delete index if already exists
index_name = 'langchain-retrieval-augmentation-fast'
# check if index exists
existing_indexes = pc.list_indexes().names()

if index_name not in existing_indexes:
    # create a new index if it does not exist
    pc.create_index(
        index_name,
        dimension=1536,  # dimensionality of text-embedding-ada-002
        metric='dotproduct',
        spec=spec
    )
    print(f"Index '{index_name}' created.")
else:
    print(f"Index '{index_name}' already exists.")

# connect to the index
index = pc.Index(index_name)

# describe index stats
index.describe_index_stats()

# only upsert documents if the index was just created
if index_name not in existing_indexes:
    for batch in dataset.iter_documents(batch_size=100):
        index.upsert(batch)
    print(f"Documents upserted into the index '{index_name}'.")

from langchain_openai import OpenAIEmbeddings

model_name = 'text-embedding-ada-002'
embed = OpenAIEmbeddings(model=model_name, openai_api_key=os.environ["OPENAI_API_KEY"])

from langchain_community.vectorstores import Pinecone

vectorstore = Pinecone(index, embed.embed_query, "text")

from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# completion llm
llm = ChatOpenAI(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    model_name='gpt-3.5-turbo',
    temperature=0.0
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)


def chatbot_response(user_input):
    response = qa.invoke(user_input)
    return response['result']


# Example usage:
user_input = input("You: ")
while user_input.lower() != 'bye':
    bot_response = chatbot_response(user_input)
    print("Bot:", bot_response)
    user_input = input("You: ")