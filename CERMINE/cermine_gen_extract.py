import csv
import sys
import os
import shutil
from os import path
from pathlib import Path
from shutil import copy
from os.path import splitext
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PdfAct.pdfact_extract import load_data, load_all_data, crop_pdf, compute_metrics, compute_results
from cermine_parse_xml import extract_cermine_metadata, parse_para_cermine, parse_sec_cermine, parse_ref_cermine, parse_metadata_cermine
from GROBID.grobid_fulltext_extr import create_gt_df, get_gt_metadata
from Refextract.refextract_extract import sort_files

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

def create_gt_df_nonsubset(dir, label):
    PDFlist=load_data(dir, label)
    IDs=[]
    values=[]
    pageno=[]
    for pdf in PDFlist:
        gt_df=get_gt_metadata(pdf, '',label, False)
        if isinstance(gt_df, type(None)):
            continue
        data_str = gt_df['token'].astype(str).str.cat(sep=' ')
        values.append(data_str)
        id=os.path.splitext(pdf.pdf_name)[0]
        IDs.append(id)
        pageno.append(pdf.page_number)
    gt= label + '_gt'
    data_gt_df=pd.DataFrame({'ID':IDs, gt:values, 'page':pageno})
    return data_gt_df


def sort_label_files(dir, label):
    PDFlist = load_data(dir, label)
    p = Path(dir + "/sorted_pdfs/")
    p.mkdir(parents=True, exist_ok=True)
    for PDF in PDFlist:
        get_gt_metadata(PDF, p,label, True)
    return str(p)

def build_gt_df(dir, label):
    """
    Build a ground-truth DataFrame for `label` over all PDFs in `dir`.
    """
    pdf_list = load_data(dir, label)
    records = []
    for pdf in pdf_list:
        gt = get_gt_metadata(pdf, '', label, False)
        if gt is None:
            continue
        text = gt['token'].astype(str).str.cat(sep=' ')
        records.append({
            'ID': os.path.splitext(pdf.pdf_name)[0],
            f'{label}_gt': text,
            'page': pdf.page_number,
        })
    return pd.DataFrame(records)


def main():
    base_dir = './Data/Docbank_sample'
    output_dir = './Data/results/Cermine'
    os.makedirs(output_dir, exist_ok=True)

    labels = ['title', 'abstract', 'author', 'paragraph', 'section', 'reference']

    all_results = []

    # loop over each label, extract, parse, merge, compute
    for label in labels:
        print(f"Processing label: {label}")
        # 1. select only PDFs with this label
        workdir = sort_files(base_dir, label)

        # 2. run Cermine to produce .cermxml files
        extract_cermine_metadata(workdir)

        # 3. parse Cermine XML output
        extracted = parse_map[label](workdir)
        # rename extracted column for consistency
        col = extracted.columns[1]
        extracted = extracted.rename(columns={col: f'{label}_ex'})

        # 4. ground-truth
        gt_df = build_gt_df(workdir, label)

        # 5. merge on ID & page
        if label in metadata_labels:
            merged = pd.merge(
                extracted.astype(str),
                gt_df.astype(str),
                on='ID'
            )
        else:
            merged = pd.merge(
                extracted.astype(str),
                gt_df.astype(str),
                on=['ID', 'page']
            )

        # 6. compute metrics
        for _, row in tqdm(merged.iterrows(), total=len(merged), desc=f"Computing {label}"):
            ex_text = row[f'{label}_ex']
            gt_text = row[f'{label}_gt']
            temp = pd.DataFrame([{label: ex_text, f'{label}_gt': gt_text}])
            f1, prec, rec, dist = compute_results(temp, label)
            all_results.append([
                'CERMINE',
                row['ID'] + '.pdf',
                row['page'],
                label,
                prec,
                rec,
                f1,
                dist,
            ])

    # 7. write single results file
    cols = ['Tool', 'ID', 'Page', 'Label', 'Precision', 'Recall', 'F1', 'SpatialDist']
    results_df = pd.DataFrame(all_results, columns=cols)
    results_df.to_csv(os.path.join(output_dir, 'Cermine_all_labels.csv'), index=False)

if __name__ == '__main__':
    main()
