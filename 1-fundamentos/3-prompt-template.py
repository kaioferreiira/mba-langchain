from langchain.prompts import PromptTemplate

template = PromptTemplate(
    input_variables=["name"],
    template="Olá, {name}! Bem-vindo ao curso de LangChain!"
)

text = template.format(name="Kaio")
print(text)