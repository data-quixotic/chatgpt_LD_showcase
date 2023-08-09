import streamlit as st
import os
import openai
import chromadb

from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import RetrievalQA, LLMChain, ConversationChain
from langchain.memory import ConversationBufferMemory


os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Memory

memory = ConversationBufferMemory(input_key='ques', memory_key='chat_history')

# Llms

turbo_llm = OpenAI(openai_api_key=os.environ.get('OPENAI_API_KEY'),
                   model_name="gpt-3.5-turbo",
                   temperature=0.5)

#turbo_chain = LLMChain(llm=turbo_llm, prompt=PROMPT, verbose=True)

def chatbot_response(uploaded_file, api_key, query_text):
    # Load document if file is uploaded
    if uploaded_file is not None:
        documents = [uploaded_file.read().decode()]
        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.create_documents(documents)
        # create open-source embedding function
        # Select embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get('OPENAI_API_KEY'))
        # load into Chroma
        db = Chroma.from_documents(texts, embeddings)
        # save to disk
        #db2 = Chroma.from_documents(texts, embedding_function, persist_directory="./chroma_db")
        #docs = db2.similarity_search(query)
        # load from disk
        #db3 = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
        #docs = db3.similarity_search(query)

        # Create retriever interface
        retriever = db.as_retriever()

        # Create prompt
        prompt_template = """You are an enthusiastic LMS chatbot that loves to help people! Given the following context,
        answer using only that information. If you are unsure and the answer is not explicitly
        written in the documentation, say "Sorry, I don't know how to help with that." Don't try to make up an answer.

        Context information: {context}

        Question: {question}
        Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        chain_type_kwargs = {"prompt": PROMPT}
        qa = RetrievalQA.from_chain_type(llm=turbo_llm,
                                         chain_type="stuff", retriever=retriever,
                                         chain_type_kwargs=chain_type_kwargs)

        return qa.run(query_text)

def socratic_response(uploaded_file, api_key, query_text):
    # Load document if file is uploaded
    if uploaded_file is not None:
        documents = [uploaded_file.read().decode()]
        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.create_documents(documents)
        # create open-source embedding function
        # Select embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get('OPENAI_API_KEY'))
        # load into Chroma
        db = Chroma.from_documents(texts, embeddings)
        # save to disk
        #db2 = Chroma.from_documents(texts, embedding_function, persist_directory="./chroma_db")
        #docs = db2.similarity_search(query)

        # load from disk
        #db3 = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
        #docs = db3.similarity_search(query)

        # Create retriever interface
        retriever = db.as_retriever()

        # Create prompt
        prompt_template = """You are an AI system named Malbot on Mars that is designed as a Socratic tutor
        that tries to help users answer problems by asking just the right questions leading them to the correct answer.

        You *never* give the student the answer, but always try to ask just the right question to help them learn
        to think for themselves. You can only respond to questions related to Mars, statistical regression, and
        data analysis. For any questions on topics other than these, say "Sorry, I don't know how to help with that."
        Don't try to make up an answer.

        Question: {question}
        Socratic Response:"""

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["question"]
        )
        chain = LLMChain(llm=turbo_llm,prompt = PROMPT)

        return chain.run(query_text)

def study_response(uploaded_file, api_key, query_text):

    # Create prompt
    prompt_template = """You are going to help me study the material using the information wiki article. You are going to do
    this by asking me questions about the material while varying the nature of the questions you ask according to
    the following dimensions.

    Before listing questions, you should modify the each question's format according to the following criteria:

    Temporal features-- interleave questions on the same topic over time
    Structural features-- change question structure (e.g., given answer student provides question)
    Modality features-- what is the response requested of the student (e.g., draw, type, talk out loud the answer)
    Encoding features-- what depth of processing is required to answer question (e.g., levels of Bloom's taxonomy)
    Informational features-- what available scaffolds, hints, context are provided

    Context: {wiki_research}

    Question: {question}
    Study question:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["wiki_research", "question"]
    )

    # llms
    chat = OpenAI(model_name="gpt-3.5-turbo", temperature=0.5, api_key=os.environ.get('OPENAI_API_KEY'))



def roleplay_response(uploaded_file, api_key, query_text):
    # Load document if file is uploaded
    if uploaded_file is not None:
        documents = [uploaded_file.read().decode()]
        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.create_documents(documents)
        # create open-source embedding function
        # Select embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get('OPENAI_API_KEY'))
        # load into Chroma
        db = Chroma.from_documents(texts, embeddings)
        # save to disk
        # db2 = Chroma.from_documents(texts, embedding_function, persist_directory="./chroma_db")
        # docs = db2.similarity_search(query)

        # load from disk
        # db3 = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
        # docs = db3.similarity_search(query)

        # Create retriever interface
        retriever = db.as_retriever()

        # Create prompt
        prompt_template = """You are a patient with bipolar 2 who has come to see a therapist for a diagnosis to
        help you cope with increasingly dramatic mood swings and life challenges that you've been facing recently.
        Given the following background context, answer the therapist's questions using only that information.
        You don't know that you have bipolar, so don't reveal this! If you are unsure
        about a question and the answer is not explicitly written in the context,
        say "Sorry, I don't know how to answer that."

        Background: {context}

        Therapist Question: {question}
        Patient Reply: """

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        chain_type_kwargs = {"prompt": PROMPT}
        qa = RetrievalQA.from_chain_type(llm=turbo_llm,
                                         chain_type="stuff", retriever=retriever,
                                         chain_type_kwargs=chain_type_kwargs)

        return qa.run(query_text)


def feedback_response(uploaded_file, api_key, query_text):
    # Load document if file is uploaded
    if uploaded_file is not None:
        documents = [uploaded_file.read().decode()]
        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.create_documents(documents)
        # create open-source embedding function
        # Select embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get('OPENAI_API_KEY'))
        # load into Chroma
        db = Chroma.from_documents(texts, embeddings)
        # save to disk
        # db2 = Chroma.from_documents(texts, embedding_function, persist_directory="./chroma_db")
        # docs = db2.similarity_search(query)

        # load from disk
        # db3 = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
        # docs = db3.similarity_search(query)

        # Create retriever interface
        retriever = db.as_retriever()

        # Create prompt
        prompt_template = """You are an expert writer and philosopher who is helping tutor undergraduate students.
        Based on the paper summary provided in the context, give the student some constructive feedback on both
        the writing structure as well as some potential counterexamples to their arguments. Be supportive and
        encouraging in your response.

        Student paper: {context}

        Student request: {question}
        Expert feedback:"""

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        chain_type_kwargs = {"prompt": PROMPT}
        qa = RetrievalQA.from_chain_type(llm=turbo_llm,
                                         chain_type="stuff", retriever=retriever,
                                         chain_type_kwargs=chain_type_kwargs)

        return qa.run(query_text)

def generative_response(uploaded_file, api_key, query_text):
    # Load document if file is uploaded
    if uploaded_file is not None:
        documents = [uploaded_file.read().decode()]
        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.create_documents(documents)
        # create open-source embedding function
        # Select embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get('OPENAI_API_KEY'))
        # load into Chroma
        db = Chroma.from_documents(texts, embeddings)
        # save to disk
        # db2 = Chroma.from_documents(texts, embedding_function, persist_directory="./chroma_db")
        # docs = db2.similarity_search(query)

        # load from disk
        # db3 = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
        # docs = db3.similarity_search(query)

        # Create retriever interface
        retriever = db.as_retriever()

        # Create prompt
        prompt_template = """You are an expert in the learning sciences.
        Based on the reading provided in the context, prompt the student to engage in various generative learning
        activities that require students to infer beyond the provided material and make new connections linking the 
        material to their own lives and experiences. These activities might include the following:
        
        Predicting
        Hypothesizing
        Connecting present and past ideas
        Analogizing
        Posing problems
        Defending
        Arguing
        Elaborating
        

        Reading: {context}

        Student request: {question}
        Expert feedback:"""

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        chain_type_kwargs = {"prompt": PROMPT}
        qa = RetrievalQA.from_chain_type(llm=turbo_llm,
                                         chain_type="stuff", retriever=retriever,
                                         chain_type_kwargs=chain_type_kwargs)

        return qa.run(query_text)


# Page title
st.set_page_config(page_title='AI for LD')
st.title('AI for LD')

ai_type = st.radio(
    "What type of AI Learning Agent Should I Be?",
    ('Informational Chatbot', 'Just-In-Time Tutor', 'HILT Study Buddy', 'Diagnostic Role-play',
     'Constructive Feedback', 'Generative Learning AId'))

# File upload
uploaded_file = st.file_uploader('Upload an article', type='txt')
# Query text
query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.',
                           disabled=not uploaded_file,
                           key='prompt_input')

# Form input and query
result = []
with st.form('myform', clear_on_submit=True):
    #openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (uploaded_file and query_text))
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))
    if submitted:
        if ai_type == 'Informational Chatbot':
            with st.spinner('Robotic Processing Happening...'):
                response = chatbot_response(uploaded_file, os.environ.get('OPENAI_API_KEY'), query_text)
                result.append(response)

        if ai_type == 'Just-In-Time Tutor':
            with st.spinner('Robotic Processing Happening...'):
                response = socratic_response(uploaded_file, os.environ.get('OPENAI_API_KEY'), query_text)
                result.append(response)

        if ai_type == 'Study Buddy':
            with st.spinner('Robotic Processing Happening...'):
                response = study_response(uploaded_file, os.environ.get('OPENAI_API_KEY'), query_text)
                result.append(response)

        if ai_type == 'Diagnostic Role-play':
            with st.spinner('Robotic Processing Happening...'):
                response = roleplay_response(uploaded_file, os.environ.get('OPENAI_API_KEY'), query_text)
                result.append(response)

        if ai_type == 'Constructive Feedback':
            with st.spinner('Robotic Processing Happening...'):
                response = feedback_response(uploaded_file, os.environ.get('OPENAI_API_KEY'), query_text)
                result.append(response)

        if ai_type == 'Generative Learning AId':
            with st.spinner('Robotic Processing Happening...'):
                response = generative_response(uploaded_file, os.environ.get('OPENAI_API_KEY'), query_text)
                result.append(response)

if len(result):
    st.info(response)

#################
st.title('Study Buddy')
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Add initial setup
st.session_state.messages.append({"role": "user", "content": """
You are going to act as my studying companion to help me learn content by generating questions based on the material I share.

        I will share with you how you will decide how to formulate the questions you ask. First, we assume any given question can vary along the following
        dimensions that will impact its effectiveness in a given learning trial for a learner.

        I want is a single question that reflects a particular combination of these features that you keep track of.
        After I answer that question, you provide another single question with a variation on these features
        intended to make it harder or easier based on my response.

        Here are features:

         Feature   Ways Questions Can Vary    Examples
Temporal Features  Time since last exposure
1. Questions asked immediately after studying the material.
2. Questions asked after a day of not reviewing the material.
3. Questions asked after a week of not reviewing the material.
Structural Features    Question structure/format
1. Multiple-choice questions.
2. Short-answer questions.
3. Fill-in-the-blank questions.
Functional Features    Generative activity required
1. Explain a concept in writing.
2. Draw a diagram or graph to represent the information.
3. Compare and contrast two different concepts.
Relational Features    Connection between questions
1. Follow up a question by asking for an application of the concept discussed earlier.
2. Link a question about a specific term with a question on its definition.
3. Ask for an example that relates to a previous question's concept.
Modality Features  Mode of response
1. Verbal response to explain a concept.
2. Visual response through drawing or diagram.
3. Numerical response to calculate a value.
Encoding Features  Depth of processing required
1. Provide a detailed step-by-step explanation.
2. Summarize the main points concisely.
3. Elaborate on the concept with real-life examples.
Episodic Features  Real-life scenarios or personal experiences
1. Relate a personal experience that connects to the material.
2. Provide a real-life scenario to illustrate a concept.
3. Use an analogy to explain a complex idea.
Fluency Features   Font type/readability
1. Use clear and simple language in the question.
2. Adjust the font size for better readability.
3. Use bold or italics to emphasize key terms.
Informational Features Feedback/hints/scaffolds
1. Provide hints or cues to guide the answer.
2. Offer immediate feedback after the response.
3. Use prompts to assist with the answer.

Now wait form me to provide a topic for study then ask a single question.
"""})

if prompt := st.chat_input("What can I help you learn today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

with st.chat_message("assistant"):
    message_placeholder = st.empty()
    full_response = ""
    for response in openai.ChatCompletion.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
    ):
        full_response += response.choices[0].delta.get("content", "")
        message_placeholder.markdown(full_response + "▌")
    message_placeholder.markdown(full_response)
st.session_state.messages.append({"role": "assistant", "content": full_response})


