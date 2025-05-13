import pdfplumber

with pdfplumber.open("./EvaluationHealthTest.pdf") as pdf:
    for page in pdf.pages :
        words = page.extract_words()
        for word in words:
            print(word['text'])