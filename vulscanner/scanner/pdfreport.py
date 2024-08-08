# importing modules 
from reportlab.pdfgen import canvas 
from reportlab.pdfbase.ttfonts import TTFont 
from reportlab.pdfbase import pdfmetrics 
from reportlab.lib import colors 
  


def generate_report(file_name,df):
    # initializing variables with values 
    fileName = 'VulnerabilityDetectionReport.pdf'
    documentTitle = file_name +'__FileReport'
    title = 'CWE REPORT ' 
    subTitle = "File Name: "+ file_name
    textLines = [ 
        'Vulnerability Detection Module',
        'Our System analyzed the feeded file',
        'We analyzed total of ' + str(len(df))+ ' functions in the file',
        
        
            
    ] 
    # image = 'image.jpg'
    
    # creating a pdf object 
    pdf = canvas.Canvas(fileName) 
    
    # setting the title of the document 
    pdf.setTitle(documentTitle) 
    
    # registering a external font in python 
    # pdfmetrics.registerFont( 
    #     TTFont('abc', 'SakBunderan.ttf') 
    # ) 
    pdfmetrics.registerFont(TTFont('TNR', 'times.ttf'))
    
    # creating the title by setting it's font  
    # and putting it on the canvas 
    # pdf.setFont('abc', 36) 
    pdf.drawCentredString(300, 770, title) 
    
    # creating the subtitle by setting it's font,  
    # colour and putting it on the canvas 
    pdf.setFillColorRGB(0, 0, 0) 
    pdf.setFont("Courier-Bold", 24) 
    pdf.drawCentredString(290, 720, subTitle) 
    
    # drawing a line 
    pdf.line(30, 710, 550, 710) 
    
    # creating a multiline text using  
    # textline and for loop 
    text = pdf.beginText(40, 680) 
    text.setFont("Courier", 18) 
    text.setFillColor(colors.black) 
    for line in textLines: 
        text.textLine(line) 
    pdf.drawText(text) 
    
    # drawing a image at the  
    # specified (x.y) position 
    # pdf.drawInlineImage(image, 130, 400) 
    
    # saving the pdf 
    pdf.save() 