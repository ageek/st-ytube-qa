from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


#creating a vectorestore embedding for given video url
def creating_db(video_url, embeddings):
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    docs = text_splitter.split_documents(transcript)

    '''
    when a user asks a question, this database will be used to perform the similarity search and
    generate output based on that
    '''
    db=Chroma.from_documents(docs, embedding=embeddings)

    return db

#get response
def get_response(db, query, openai_api_key, k=5):
    '''
    gpt-3.5 turbo can handle up to 4097 tokens. Setting the chunksize to 1000 and k to 4 maximizes
    the number of tokens to analyze.
    '''
    
    docs = db.similarity_search(query, k)
    
    docs_page_content = " ".join([d.page_content for d in docs])
    
    chat = ChatOpenAI(
            openai_api_key=openai_api_key,
            temperature=.4)
    
    #tempalte
    template="""
    You are a helpful assistant who can answer question from Youtube videos based on the video's transcript: {docs}
    Only use the factual information from transcript to answer the question.
    Do not try to make up an answer if you dont have the corresponding datato answer. 
    If you feel like you don't have enough information to answer the question, say: "Sorry, I cannot answer that".
    Your answer should be concise and not more than 3 sentences.
    """
    
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    
    #human question prompt
    human_template='Answer the following question: {question}'
    
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    
    #chaining
    chain = LLMChain(llm=chat, prompt=chat_prompt, verbose=True)
    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    
    return response, docs

