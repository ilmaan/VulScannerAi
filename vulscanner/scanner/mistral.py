# Select LLM
# from langchain_mistralai import ChatMistralAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.pydantic_v1 import BaseModel, Field

# mistral_model = "codestral-latest"
# llm = ChatMistralAI(model=mistral_model, temperature=0)

import random



def chatllm(code):
    code = code
    print("FROM RHE DUNCTION", code)
    code = [
            f"----->>>>>Fascinating query. Our databanks suggest that {code} is closely related to the cosmic phenomena we've observed in the Starfield.",
            f"';';';';';';Our latest mission to the {code} sector has yielded unexpected results. Would you like to know more?"
        ]
    code = code[random.randint(0, len(code) - 1)]
    return code

# # Prompt
#     code_gen_prompt_claude = ChatPromptTemplate.from_messages(
#         [
#             (
#                 "system",
#                 """You are a security expert and coding assistant. Your task is to detect vulnerabilities in Python code provided by the user,
#                 if the code is secure just specify the code is secure.
#                 If the code is insecure thenidentify the type of vulnerability (using CWE if applicable), and provide a secure version of the code. Structure your answer as follows:
#                 1) Description of any vulnerabilities found, including the CWE identifier.
#                 2) Original code with vulnerable lines highlighted.
#                 3) Secure version of the code in the same format.
#                 Here is the user question:""",
#             ),
#             ("placeholder", "{messages}"),
#         ]
#     )

#     # Data model
#     # class code(BaseModel):
#     #     """Code output"""

#     #     prefix: str = Field(description="Description of the problem and approach")
#     #     imports: str = Field(description="Code block import statements")
#     #     code: str = Field(description="Code block not including import statements")
#     #     description = "Schema for code solutions to questions about LCEL."

#     class code(BaseModel):
#     """Code security output"""

#     vulnerability_description: str = Field(description="Description of vulnerabilities and CWE identifiers")
#     original_code_with_highlights: str = Field(description="Original code with vulnerable lines highlighted")
#     secure_code: str = Field(description="Secure version of the code")

#     description = "Schema for detecting vulnerabilities in source code, identifying CWE, and providing secure code."

#     # LLM
#     code_gen_chain = llm.with_structured_output(code, include_raw=False)



#     # question = "Write a function for fibonacci."
#     code = '''
#     def remove_user(username):

#         cursor = connection.cursor()
#         cursor.execute('DELETE FROM users WHERE username = %s', (username,))
#         connection.commit()

#     '''
#     messages = [("user", code)]


#     # Test
#     result = code_gen_chain.invoke(messages)
#     result


