from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import trim_messages
from langchain_core.runnables import RunnableLambda

load_dotenv()

#TODO: Ajustar para pegar prompt de uma pasta externa
# Criar um agent específico para ADRs

prompt = ChatPromptTemplate.from_messages([
    ("""system", "Você é um assistente de criação de ADRs (Arquitetura de Decisão de Registro). 
     Sempre responda com um ADR completo e bem estruturado com base na entrada do usuário.
    Mantenha o formato consistente e inclua todos os detalhes relevantes."""),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])

llm = ChatOpenAI(model="gpt-5-nano", temperature=0.9)

def prepare_inputs(payload: dict) -> dict:
    raw_history = payload.get("raw_history", [])
    trimmed = trim_messages(
        raw_history,
        token_counter=len,
        max_tokens=2,
        strategy="last",
        start_on="human",
        include_system=True,
        allow_partial=False,
    )
    return {"input": payload.get("input",""), "history": trimmed}

prepare = RunnableLambda(prepare_inputs)
chain = prepare | prompt | llm

session_store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]


conversational_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="raw_history"
)

config = {"configurable": {"session_id": "demo-session"}}

# Interactions
response1 = conversational_chain.invoke({"input": "Ola, vou explocar o contexto da minha iniciativa, ela se chama iniciativa homem de ferro"}, config=config)
print("Assistant: ", response1.content)
print("-"*30)

response2 = conversational_chain.invoke({"input": "pode repetir o nome da iniciativa?"}, config=config)
print("Assistant: ", response2.content)
print("-"*30)

response3 = conversational_chain.invoke({"input": "pode repetir o nome da iniciativa dizendo oque ela faz?"}, config=config)
print("Assistant: ", response3.content)
print("-"*30)
