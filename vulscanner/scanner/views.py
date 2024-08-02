from django.shortcuts import render

from django.http import HttpResponse


from django.views import View
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
            # Process the file data
            # else:
            #     print("No file sent")
            return render(request, 'file_scanner.html')   

    def get(self,request):
        
        return render(request,'file_scanner.html')Vul