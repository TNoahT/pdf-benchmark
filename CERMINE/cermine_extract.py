import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PdfAct.pdfact_extract import load_data, load_all_data, crop_pdf, compute_metrics
from cermine_parse_xml import extract_cermine_metadata, parse_para_cermine, parse_sec_cermine, parse_ref_cermine, parse_metadata_cermine

metadata_labels = ['title', 'abstract', 'author']

# Mapping for CERMINE functions
parse_map = {
    'title':    lambda d: parse_metadata_cermine(d)[['ID', 'Title']].rename(columns={'Title': 'title_ex'}),
    'abstract': lambda d: parse_metadata_cermine(d)[['ID', 'Abstract']].rename(columns={'Abstract': 'abstract_ex'}),
    'author':   lambda d: parse_metadata_cermine(d)[['ID', 'Authors']].rename(columns={'Authors': 'author_ex'}),
    'paragraph': parse_para_cermine,
    'section':   parse_sec_cermine,
    'reference': parse_ref_cermine,
}

def extract_cermine(base_dir, labels):
    resultdata = []
    
    for label in labels:
        PDFList = load_data(base_dir, label)

        for pdf in PDFList:
            pdfpath = pdf.filepath + os.sep + pdf.pdf_name
            croppedfile = crop_pdf(pdf.filepath, pdf.pdf_name, pdf.page_number)

            if isinstance(croppedfile, type(None)):
                continue

            # Assuming parse_map is a dictionary of functions to extract data
            if label in parse_map:
                extracted_data = parse_map[label](croppedfile)
                resultdata.append([pdf.pdf_name, pdf.page_number, label, extracted_data])

    resultsdf = pd.DataFrame(resultdata, columns=['PDF Name', 'Page Number', 'Label', 'Extracted Data'])
    return resultsdf

def main():
    base_dir = './Data/Docbank_sample'

    labels = ['title', 'abstract', 'author', 'paragraph', 'section', 'reference']

    resultsdf = extract_cermine(base_dir, labels)
    filename = 'cermine_all.csv'
    
    outputf = './Data/results/Cermine/' + filename
    os.makedirs(outputf, exist_ok=True)
    resultsdf.to_csv(outputf, index=False)

if __name__ == "__main__":
    main()