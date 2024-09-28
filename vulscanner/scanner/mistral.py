# Select LLM
# from langchain_mistralai import ChatMistralAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.pydantic_v1 import BaseModel, Field

# mistral_model = "codestral-latest"
# llm = ChatMistralAI(model=mistral_model, temperature=0)


import pandas as pd
import re
from typing import List
from pydantic import BaseModel, Field

import random

from scanner.pdfreport import generate_report

from scanner.pdfrep import PDFPSReporte

from scanner.gputest import chatcomp, chatcompfile




def chatllm(code):
    code = code
    print("FROM RHE DUNCTION", code)
    

    resp = chatcomp(code)

    # response = resp

    # # Extract CWE ID
    # cwe_id = re.search(r'CWE-id:\s*(.*)', response)
    # cwe_id = cwe_id.group(1) if cwe_id else None

    # # Extract Reason
    # reason = re.search(r'Reason:\s*(.*?)(?=\n\n)', response, re.DOTALL)
    # reason = reason.group(1).strip() if reason else None

    # # Extract Secure Version
    # # secure_version = re.search(r'Secure Version:\s*\n(.*?)$', response, re.DOTALL)
    # # secure_version = secure_version.group(1).strip() if secure_version else None

    # # Extract Secure Version
    # # secure_version = re.search(r'Secure Version:\s*\n(.*)', response, re.DOTALL)
    # # secure_version = secure_version.group(1).strip() if secure_version else None
    # # Find the start of the Secure Version section
    # start_index = response.find("Secure Version:")

    # if start_index != -1:
    #     # If "Secure Version:" is found, extract everything after it
    #     secure_version = response[start_index + len("Secure Version:"):].strip()
    # else:
    #     secure_version = None



    # # Print the extracted information
    # print("\n\n\nCWE ID-------------------------------------------------------------------------------:", cwe_id)
    # print("\nReason------------------------------:", reason)
    # print("\nSecure Version------------------------------------:")
    # print(secure_version)



    return resp

# Prompt
def prompt():
    code_gen_prompt_claude = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a security expert and coding assistant. Your task is to detect vulnerabilities in Python code provided by the user,
                if the code is secure just specify the code is secure.
                If the code is insecure thenidentify the type of vulnerability (using CWE if applicable), and provide a secure version of the code. Structure your answer as follows:
                1) Description of any vulnerabilities found, including the CWE identifier.
                2) Original code with vulnerable lines highlighted.
                3) Secure version of the code in the same format.
                Here is the user question:""",
            ),
            ("placeholder", "{messages}"),
        ]
    )

    # Data model
    # class code(BaseModel):
    #     """Code output"""

    #     prefix: str = Field(description="Description of the problem and approach")
    #     imports: str = Field(description="Code block import statements")
    #     code: str = Field(description="Code block not including import statements")
    #     description = "Schema for code solutions to questions about LCEL."

    class code(BaseModel):
        """Code security output"""

        vulnerability_description: str = Field(description="Description of vulnerabilities and CWE identifiers")
        original_code_with_highlights: str = Field(description="Original code with vulnerable lines highlighted")
        secure_code: str = Field(description="Secure version of the code")

        description = "Schema for detecting vulnerabilities in source code, identifying CWE, and providing secure code."

        # LLM
        code_gen_chain = llm.with_structured_output(code, include_raw=False)



        # question = "Write a function for fibonacci."
        code = '''
        def remove_user(username):

            cursor = connection.cursor()
            cursor.execute('DELETE FROM users WHERE username = %s', (username,))
            connection.commit()

        '''
        messages = [("user", code)]


        # Test
        result = code_gen_chain.invoke(messages)
        result


        dataset = pd.read_csv("/content/reshaped_dataset.csv")
        dataset = Dataset.from_pandas(custom_data)

        # dataset[1]['input']

        for i in range(5):
            question = dataset[i]['input']
            print("ORINGINAL CODE\n\n",question,'\n\nOUTPUT--',dataset[i]['output'])
            messages = [("user", question)]
            result = code_gen_chain.invoke(messages)
            print(result)
            print('\n\n------------------------NEW CODE---------------------------\n\n')

        dataset = pd.read_csv("/content/reshaped_dataset.csv")
        dataset = Dataset.from_pandas(custom_data)

        # dataset[1]['input']

        for i in range(5):
            question = dataset[i]['input']
            print("ORINGINAL CODE\n\n",question,'\n\nOUTPUT--',dataset[i]['output'])
            messages = [("user", question)]
            result = code_gen_chain.invoke(messages)
            print(result)
            print('\n\n------------------------NEW CODE---------------------------\n\n')




# def extract_functions_from_file(file_content: str) -> List[dict]:
#     # with open(file_path, 'r') as file:
#     #     print("FILE FROM HELL--->>>>",file_path)
#     # file_content = file.read()
#     # print("\n\nFILE FROM DESSSSSS----->>>>",file_content)

#     function_pattern = re.compile(r'(def \w+\(.*?\)):\n((?:    .*?\n)*)', re.DOTALL)
#     functions = function_pattern.findall(file_content)

#     function_list = []
#     for func_signature, func_body in functions:
#         full_function = f'{func_signature}:\n{func_body.strip()}'
#         function_list.append({
#             "function_name": func_signature,
#             "function_code": full_function
#         })

#         print("FUNCTION LS+IST------------------------------<><><><><><\n\n")
#     return function_list


def extract_functions_from_file(file_content: str) -> List[dict]:
    # Improved regex pattern to capture entire functions
    function_pattern = re.compile(r'def\s+(\w+)\s*\((.*?)\)\s*:(.*?)(?=\n(?:def|$))', re.DOTALL)
    functions = function_pattern.findall(file_content)
    function_list = []
    
    for func_name, func_args, func_body in functions:
        full_signature = f"def {func_name}({func_args}):"
        full_function = f"{full_signature}\n{func_body.strip()}"
        function_list.append({
            "function_name": full_signature,
            "function_code": full_function
        })
    print("FUNCTION LS+IST------------------------------<><><><><><\n\n",function_list)
    return function_list

# Function to analyze the functions and determine their vulnerability status
def analyze_functions(function_list: List[dict],code_file) -> pd.DataFrame:
    data = []

    for func in function_list:
        # messages = [("user", func['function_code'])]
        # result = code_gen_chain.invoke(messages)

        status = "Non-vulnerable"
        cwe_id = []
        # vul_dsc = '''The code is vulnerable to SQL Injection . This occurs when user input is not  CWE-53 properly sanitized  CWE-23 CWE 089 before being used in a SQL query. In this case, the 'username' variable is directly inserted into the SQL query, which allows an attacker to manipulate the query and potentially delete any user from the database'''

        # pattern = r'CWE-\d{2,3}'
        # cwe_id = re.findall(pattern, vul_dsc)
        # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^",cwe_id)

        # if result and hasattr(result, "vulnerability_description") and result.vulnerability_description:
        #     status = "Vulnerable"

        data.append([func['function_name'], func['function_code'], status,cwe_id])

    df = pd.DataFrame(data, columns=['Function Name', 'Function Code', 'Status','CWE IDs'])

    print(len(df))
    for f in range(len(df)):
                func_code = df.loc[f,"Function Code"]
                print("ORIGINAL CODE ____----->>>>>\n\n\n",func_code)
                respllm = chatcompfile(func_code)
                status = respllm.get("Status")
                cwe_id = respllm.get("CWE_ID")
                reason = respllm.get("Reason")
                secure_code = respllm.get("Secure Code")


                print("Status>>>>>>>>>>>>>>>>>>>>-----------:", status)
                print("CWE_ID-------------:", cwe_id)
                print("Reason------------:", reason)
                print("Secure Code-------------:", secure_code)

                # if secure_version == None:
                #     secure_version = respllm

                # messages = [("user", func_code)]
                # result = code_gen_chain.invoke(messages)
                # df.loc[f,"Status"] = '''The code is vulnerable to SQL Injection . This occurs when user input is not  CWE-53 properly sanitized  CWE-23 CWE 089 before being used in a SQL query. In this case, the 'username' variable is directly inserted into the SQL query, which allows an attacker to manipulate the query and potentially delete any user from the database'''
                df.loc[f,"Status"] = reason
                pattern = r'CWE-\d{2,3}'
                # cwe_id = re.findall(pattern, str(df.loc[f,"Status"])) 
                
                df.loc[f,"CWE IDs"] = cwe_id

                df.loc[f,"Secure Code"] = secure_code

                

                
                print('\n\n--------++++++++-----------NEW CODE----++++++++++++++++++++-----\n\n')

    # x = generate_report(code_file,df)
    # vlist = vul_list(df)
    vlist = 3
    print('---------------------',vlist)

    # vdet = vul_details(vlist)
    vdet = "vul_details(vlist)"
    print('------------------>>>>>>----',vdet)

    

    report = PDFPSReporte('psreereportnewtodd.pdf',code_file,df,vdet)



    print("REPORT--------------------",report,type(report))


    
    return df


def vul_list(df):
    vul_list = set()
    for i in range(len(df)):
          print("TIMESSSS",i,len(df))
          for j in df.loc[i,"CWE IDs"]:
               vul_list.add(j)
    # print(vul_list,'---------VUL LIST--------------------')
    return vul_list



def vul_details(vlist):
    vul_details = {}
    dsc = '''"Improper Neutralization of Special Elements used in an SQL Command ('SQL Injection')", allows an attacker to inject malicious SQL code into a robotic system's database, compromising its:
Data integrity and confidentiality
System operations and availability
Authentication and authorization mechanisms
This can lead to unauthorized data access, system downtime, and manipulation of the robotic system's actions or movements, posing a risk to human safety and system reliability.'''
    for vul in vlist:
        #   print(vul,'-------VULL___DEAISL')
        
          vul_details[vul] = dsc
        #   print('+++++++++++++++++++++\n\n',vul,'---',vul_details[vul],'\n\n')
    
    return vul_details
          
     
     



# Example usage
# file_path = '/content/views.py'  # Replace with your actual Python file path
# functions = extract_functions_from_file(file_path)
# df = analyze_functions(functions)

# print(df)


