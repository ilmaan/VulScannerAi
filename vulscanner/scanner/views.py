from django.shortcuts import render

from django.http import HttpResponse

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import random


from django.views import View

from scanner.mistral import chatllm, extract_functions_from_file, analyze_functions
# Create your views here.


def home(request):
    
    return render(request, 'home.html')


def file_scanners(request):
    if request.method == 'POST':
        try:
            print("IN POST")
            code_file = request.FILES.get('code_file')
            if code_file:
                print("FILEE GOT-->>",code_file)
        except Exception as e:
            print("EXCEPTION OCCURED",e)
    
    return render(request, 'file_scanner.html')


def code_scanner(request):
    
    return render(request, 'code_scanner.html')




class file_scanner(View):
    def post(self,request):
        if request.method == 'POST':
            # if 'code_file' in request.FILES:
            code_file = request.FILES.get('code_file')
            txt = request.POST.get('textdata')
            
            
            print("FILEE GOT-->>",txt,'---------------', code_file,type(code_file))
            file_content = code_file.read().decode('utf-8')
            fucntions = extract_functions_from_file(file_content)
            df = analyze_functions(fucntions)
            
            for f in range(len(df)):
                func_code = df.loc[f,"Function Code"]
                # messages = [("user", func_code)]
                # result = code_gen_chain.invoke(messages)
                print(func_code)
                print('\n\n------------------------NEW CODE---------------------------\n\n')
            # print(fucntions)
            
            # try:
            #     file_content = code_file.read().decode('utf-8')
            #     print("\n\nFILE FROM DESSSSSS----->>>>", file_content)
            # except Exception as e:
            #     print("ERROR--->>",e)    
            # Process the file data
            # else:
            #     print("No file sent")
            return render(request, 'file_scanner.html')   

    def get(self,request):
        
        return render(request,'file_scanner.html')
    

@csrf_exempt  # This decorator exempts the view from CSRF verification
def get_response(request):
    if request.method == 'POST':
        print("GOT HERE---->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        input_text = request.POST.get('input')
        print("INPUT TEXT",input_text)
        responses = chatllm(input_text)
        # responses = [
        #     f"----->>>>>Fascinating query. Our databanks suggest that {input_text} is closely related to the cosmic phenomena we've observed in the Starfield.",
        #     f"';';';';';';Our latest mission to the {input_text} sector has yielded unexpected results. Would you like to know more?"
        # ]
        # responses = responses[random.randint(0, len(responses) - 1)]
        print("CAME FROM FUNCT\n\n",responses)
        return JsonResponse({'response': responses})
    return JsonResponse({'error': 'Invalid request'}, status=400)