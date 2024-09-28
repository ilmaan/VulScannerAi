from transformers import pipeline
import torch

import re
import json



model_id = "c:\\Users\\cente\\Desktop\\ilmaan project\\llama3.1\\Meta-Llama-3.1-8B-Instruct"


# pipe = pipeline(
#     "text-generation",
#     model=model_id,
#     model_kwargs={"torch_dtype":torch.bfloat16},
#     device="cuda"
# )

# messages = [
#     {"role":"Tester","content":'''
#     import os

# class RobotAuthenticator:
#     def __init__(self):
#         self.auth_token = os.getenv(""ROBOT_AUTH_TOKEN"")

#     def authenticate(self):
#         print(f"Authenticating robot with token: self.auth_token")
#         # Authentication logic here...
#     Is the above code Secure or Insecure. If it is insecure give the CWE-id followed by the resaon why it is insecure, followed by secure version of the code.

#      ''',}
# ]

# output = pipe(messages,max_new_tokens=856,do_sample=False,)

# assit_resp = output[0]["generated_text"][-1]["content"]

# print(assit_resp)


def chatcomp(code):
    print('-----------------------------<><><><><><><><><',code)
    if code ==None or code=="":
        return "TEST"
    else:
        pipe = pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype":torch.bfloat16},
        device="cuda")

        messages = [
            {"role":"Tester","content":code+'''
            Is the above code Secure or Insecure. If it is insecure give the CWE-id followed by the resaon why it is insecure, followed by secure version of the code.
            ''',}
        ]

        output = pipe(messages,max_new_tokens=856,do_sample=False,)

        assit_resp = output[0]["generated_text"][-1]["content"]

        print('ASSIST REPO \n\n ---------------------\n\n',assit_resp)

        return assit_resp
    


    
def chatcompfile(code):
    print('-----------------------------<><><><><><><><><',code)
    if code ==None or code=="":
        return "TEST"
    else:
        pipe = pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype":torch.bfloat16},
        device="cuda")

        # messages = [
        #     {"role":"Tester","content":code+'''
        #     Give the response of status,CWE-ID,Reason,Secure Code in a Json Format
        #     Status:Is the above code Secure or Insecure. 
        #     CWE_ID: If it is insecure give the CWE-id or IDS 
        #     Reason:The resaon why it is insecure 
        #     Secure Code: generate secure version of the code maintaining the syntax (Only If the code is insecure otherwise give none ).
        #     ''',}
        # ]

        messages = [
            {
                "role": "Tester",
                "content": code + '''
        Please analyze the above code and provide the following information **in JSON format only**. Do not include any explanations or additional text outside the JSON.

        **Required JSON Format:**

        {
        "Status": "<Secure or Insecure>",
        "CWE_ID": "<If insecure, provide the CWE-ID or IDS; otherwise, use null>",
        "Reason": "<If insecure, provide the reason why it is insecure; otherwise, state 'Code is secure'>",
        "Secure Code": "<If insecure, provide a secure version of the code maintaining the syntax; otherwise, use null>"
        }

        **Instructions:**

        - **Output must be valid JSON.**
        - **All strings should be enclosed in double quotes.**
        - **Use `null` (without quotes) for any field that does not apply.**
        - **Do not include any text before or after the JSON.**
        '''
            }
        ]

        output = pipe(messages,max_new_tokens=856,do_sample=False,)

        assit_resp = output[0]["generated_text"][-1]["content"]

        print('ASSIST REPO \n\n ---------------------\n\n',assit_resp)
        

        try:
# Parse the JSON string into a Python dictionary
            parsed_data = json.loads(assit_resp)

            # Print the resulting dictionary
            print("Converted to Python dictionary------------->>>>>>>>>>>>>>>>>>>>:",parsed_data,'------------------------')
            for key, value in parsed_data.items():
                print(f"{key}:")
                print(value)
                print()

#         except json.JSONDecodeError as e:
#             print(f"Error decoding JSON: {e}")
#             print("Raw response string------------->>>>:")
#             print(json_string)
        
#         # Find the JSON part
#         json_match = re.search(r'```json\n(.*?)```', assit_resp, re.DOTALL)
#         print("))))))))))))))))))))))))))))))))))))))))))))))))))))))\n\n",json_match)
#         if json_match:
#             json_string = json_match.group(1)
            
#             # Remove newlines and extra spaces from the JSON string
#             json_string = re.sub(r'\s+', ' ', json_string)
#             print("<<<<<<<<<<>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<>>>>>>>>>>>\n\n",json_string,type(json_string))
            
            
#             # Parse the JSON string
#             # try:
#                 # json_data = json.loads(json_string)

            
#             try:
# # Parse the JSON string into a Python dictionary
#                 data = json.loads(json_string)

#                 # Print the resulting dictionary
#                 print("Converted to Python dictionary:")
#                 for key, value in data.items():
#                     print(f"{key}:")
#                     print(value)
#                     print()

#             except json.JSONDecodeError as e:
#                 print(f"Error decoding JSON: {e}")
#                 print("Raw response string:")
#                 print(json_string)

#             # Parse the JSON string
#         try:
#             parsed_data = parse_custom_json(json_string)

#             # Extract the desired information into separate variables
            status = parsed_data.get("Status")
            cwe_id = parsed_data.get("CWE_ID")
            reason = parsed_data.get("Reason")
            secure_code = parsed_data.get("Secure Code")

            # Print the extracted information to verify
            # print("Status-----------:", status)
            # print("CWE_ID-------------:", cwe_id)
            # print("Reason------------:", reason)
            # print("Secure Code-------------:", secure_code)
        except Exception as e:
            print("ERROR",e)

        return parsed_data
    


# def parse_custom_json(json_string):
#     # Use regex to split the string into key-value pairs
#     pattern = r'"([^"]+)"\s*:\s*"((?:[^"\\]|\\.)*)"'
#     matches = re.findall(pattern, json_string, re.DOTALL)
    
#     # Create a dictionary from the matches
#     result = {}
#     for key, value in matches:
#         # Unescape any escaped quotes within the value
#         value = value.replace('\\"', '"')
#         result[key] = value
    
#     return result


def parse_custom_json(json_string):
    # Use regex to split the string into key-value pairs
    pattern = r'"([^"]+)"\s*:\s*"((?:[^"\\]|\\.)*)"'
    matches = re.findall(pattern, json_string, re.DOTALL)
    
    # Create a dictionary from the matches
    result = {}
    for key, value in matches:
        # Unescape any escaped quotes within the value
        value = value.replace('\\"', '"')
        
        # If the key is "Secure Code", capture everything until the end
        if key == "Secure Code":
            secure_code_pattern = r'"Secure Code"\s*:\s*"(.*?)"(?=\s*})'
            secure_code_match = re.search(secure_code_pattern, json_string, re.DOTALL)
            if secure_code_match:
                value = secure_code_match.group(1)
        
        result[key] = value
    
    return result


def extract_sections(response):
    # Extract CWE-ID
    cwe_id_pattern = r'\*\*CWE-ID:\*\*\s*(.*)\n'
    cwe_id_match = re.search(cwe_id_pattern, response)
    cwe_id = cwe_id_match.group(1) if cwe_id_match else None

    # Extract Reason
    reason_pattern = r'\*\*Reason:\*\*\s*(.*)\n'
    reason_match = re.search(reason_pattern, response)
    reason = reason_match.group(1) if reason_match else None

    # Extract Secure Code (everything after "**Secure Code:**")
    secure_code_pattern = r'\*\*Secure Code:\*\*\n```python(.*)```'
    secure_code_match = re.search(secure_code_pattern, response, re.DOTALL)
    secure_code = secure_code_match.group(1).strip() if secure_code_match else None

    return cwe_id, reason, secure_code