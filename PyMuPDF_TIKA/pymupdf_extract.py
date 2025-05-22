import os
import csv
import sys
import pandas as pd
from os import path
from tqdm import tqdm
import pymupdf as fitz

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PdfAct.pdfact_extract import load_data, load_all_data, crop_pdf, compute_metrics

tool = 'PyMuPDF'

def extract_tika(base_dir, labels):
    
    resultdata = []

    for label in labels :

        PDFList = load_data(base_dir, label)

        for pdf in tqdm(PDFList) :
            pdfpath=pdf.filepath + os.sep + pdf.pdf_name
            croppedfile=crop_pdf(pdf.filepath, pdf.pdf_name, pdf.page_number)

            output_dir = "./Data/results/Tika"
            os.makedirs(output_dir, exist_ok=True)
            file_name = os.path.splitext(os.path.basename(pdf.pdf_name))[0]
            outputfile = os.path.join(output_dir, f"{file_name}_page_{pdf.page_number}_{label}.txt")

            #print(f"[DEBUG] Active file = {croppedfile}")
            if isinstance(croppedfile,type(None)):
                continue

            doc = fitz.open(croppedfile)
            page= doc[0]
            text = page.get_text("text") # only of page 0 ?
            
            #toc = doc.get_toc()
            #print(f"[DEBUG] Table of Contents: {toc}")
            #print(f"[DEBUG] Parsed metadata: {metadata}")

            if label == 'paragraph':
                paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                with open(outputfile, 'w', encoding='utf-8') as f:
                    for para in paragraphs:
                        f.write(para + '\n\n')

            elif label == 'author':
                metadata = doc.metadata
                #print(f"[DEBUG] Metadata: {metadata}")
                author = metadata.get('author', '')
                with open(outputfile, 'w', encoding='utf-8') as f:
                    f.write(author)
            
            elif label == 'table' :
                tables = page.find_tables().tables
                if tables:
                    rows = tables[0].rows
                    with open(outputfile, 'w', encoding='utf-8') as f:
                        for row in rows:
                             #print(row)
                             if isinstance(row, (list, tuple)):
                                f.write('\t'.join(str(cell) for cell in row) + '\n')
                else:
                    with open(outputfile, 'w', encoding='utf-8') as f:
                        f.write('')

            if not os.path.isfile(outputfile) or os.path.getsize(outputfile) == 0:
                resultdata.append([tool, pdf.pdf_name, pdf.page_number, label, 0, 0, 0, 0])
                os.remove(outputfile)

            else:
                f1, pre, recall, lavsim = compute_metrics(pdf, outputfile, label)
                resultdata.append([tool, pdf.pdf_name, pdf.page_number, label, pre, recall, f1, lavsim])
                os.remove(outputfile)

            # Always clean cropped file
            if os.path.exists(croppedfile):
                os.remove(croppedfile)

    resultdf = pd.DataFrame(resultdata,
                            columns=['Tool', 'ID', 'Page', 'Label', 'Precision', 'Recall', 'F1', 'SpatialDist'])
    return resultdf

def main() :
    labels = ['table']
    base_dir = "./Data/Docbank_sample"
    resultdf = extract_tika(base_dir, labels)

    filename='PyMuPDF_ref' + '.csv'
    outputf='./Data/results/PyMuPDF/' + filename
    os.makedirs(os.path.dirname(outputf), exist_ok=True)
    resultdf.to_csv(outputf, index=False)

if __name__ == "__main__":
    main()