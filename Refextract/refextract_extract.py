import os
import sys
import shutil
from os import path
import csv
from pathlib import Path
from shutil import copy
from glob import glob
import pandas as pd
from refextract import extract_references_from_file # pip install refextract
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from GROBID.evaluate import compute_results

# TODO : fix recall > 1

class PDF:
    def __init__(self, page_number=None, pdf_name=None, filepath=None, txt_name=None, txt_data=None):
        self.page_number = page_number
        self.pdf_name = pdf_name
        self.filepath = filepath
        self.txt_name = txt_name
        self.txt_data = txt_data

def load_data(dir, label):
    """
    Create PDF objects by matching TXT files to their corresponding _black.pdf files in DocBank_sample.
    Only return PDFs that contain the specified label.
    """
    txt_files = glob(path.join(dir, "*.txt"))
    PDFlist = []

    for txt in txt_files:
        base = path.splitext(path.basename(txt))[0]  # e.g. 11.tar_1401.6921.gz_rad-lep-II-2_13
        keyword = base.rpartition("_")[0]            # e.g. 11.tar_1401.6921.gz_rad-lep-II-2
        page_number = base.split("_")[-1]
        pdf_path = path.join(dir, f"{keyword}_black.pdf")

        if path.isfile(pdf_path):
            pdf_name = path.basename(pdf_path)
            txt_name = path.basename(txt)
            txtdf = pd.read_csv(
                txt, sep='\t', quoting=csv.QUOTE_NONE, encoding='latin1',
                usecols=[0, 1, 2, 3, 4, 9],
                names=["token", "x0", "y0", "x1", "y1", "label"]
            )
            if label in txtdf["label"].values:
                PDFlist.append(PDF(page_number, pdf_name, dir, txt_name, txtdf))
        else:
            print(f"[ERROR] Cannot find expected PDF: {pdf_path}")

    #print(f"[DEBUG] Total PDFs loaded: {len(PDFlist)}")
    return PDFlist

def sort_files(dir, label):
    PDFlist = load_data(dir, 'reference')
    p = Path(dir + "/sort_pdfs/")
    p.mkdir(parents=True, exist_ok=True)
    for PDF in PDFlist:
       get_gt_ref(PDF,  p, True, label)
    return str(p)

def get_gt_ref(PDFObj, p, retflag, label):
    """
    Function has two purpose controlled by retflag parameter.
    1. retflag==True : find the GT files in DocBank containing metadata labels and copy them into seperate directory in tree called "metadata_pdfs"
    2. retflag==False: return the reference dataframes
    :param PDF: PDF Object
    :param retflag: flag to control the role of the function.
    :return: Ground truth reference dataframe.
    """
    txt_data = PDFObj.txt_data
    ref_frame_labled = txt_data.loc[(txt_data['label'] == label)]
    if len(ref_frame_labled) != 0:
        if retflag == True:
            # Extract the base name (remove last _pageNum suffix)
            base = "_".join(PDFObj.pdf_name.split("_")[:-1])
            search_pattern = os.path.join(PDFObj.filepath, f"{base}*_black.pdf")
            matches = glob(search_pattern)

            if not matches:
                print(f"[ERROR] Missing GT PDF: pattern={search_pattern}")
                return
            filename = matches[0]
            print(f"[DEBUG] Matched PDF: {filename} for TXT: {PDFObj.pdf_name}")
            # Also update the txtname path
            txtname = os.path.join(PDFObj.filepath, PDFObj.txt_name)

            # Verify existence before copying
            if not os.path.exists(filename):
                print(f"[ERROR] Missing GT PDF: {filename}")
                return
            if not os.path.exists(txtname):
                print(f"[ERROR] Missing GT TXT: {txtname}")
                return

            copy(filename, p)
            copy(txtname, p)
            return
        else:
            return ref_frame_labled

def process_for_spec_ref(gt_ref, ex_references):
    """
    This function is for refextract as using refextract we cannot extract only part of References from the PDF. It needs whole PDF for successful extraction.
    for example : if References are on page 27 and 28 in a pdf but in our dataset only 27 is label we have to handle thot by using this function.
    Here we find out the citation number from dataset and use it as a subscript to only select references which are present in the dataset.
    :param gt_ref: Ground Truth References only on specific page
    :param ex_references: Extracted References whole PDF wide
    :return: Dataframe with extracted references which are only present on specific page as in GT.
    """
    ex_ref = [x['raw_ref'] for x in ex_references]
    ex_df = pd.DataFrame(ex_ref)
    ex_df = ex_df.drop_duplicates().reset_index(drop=True)

    # num_df = gt_ref.loc[gt_ref.iloc[:, 0].str.contains(r'\[.*\]')]
    # num_df = num_df[['token']].reset_index(drop=True)
    # subscript = int(num_df['token'].iloc[0].strip('[').strip(']')) - 1
    #
    # ex_df = ex_df.loc[subscript:]

    ex_df = ex_df[0].str.cat(sep=' ')
    gt_ref = gt_ref['token'].str.cat(sep=' ')

    d = {
        'reference_gt': [gt_ref],
        'reference_ex': [ex_df],
    }
    final_df = pd.DataFrame(d)

    return final_df

def main():
    #dir_array = ['docbank_1501','docbank_1502','docbank_1503','docbank_1504', 'docbank_1505', 'docbank_1506', 'docbank_1507', 'docbank_1508',
    #             'docbank_1509', 'docbank_1510', 'docbank_1511', 'docbank_1512']
    dir_array = [""]
    for dir in dir_array:
        base_dir = "./Data/Docbank_sample" + ("/" + dir if dir else "")
        refdir = sort_files(base_dir, 'reference')
        PDFlist=load_data(refdir, 'reference')
        resultdata=[]
        for pdf in tqdm(PDFlist):
            file=pdf.filepath + os.sep + pdf.pdf_name
            if isinstance(file, type(None)):
                continue
            ex_references = extract_references_from_file(file) # For library
            gt_ref = get_gt_ref(pdf, refdir, retflag=False, label='reference')
            if len(ex_references) != 0:
                final_df=process_for_spec_ref(gt_ref, ex_references)
                f1, pre, recall, lavsim = compute_results(final_df, 'reference')
                # similarity_df, no_of_gt_tok, no_of_ex_tok, ef_ex, lavsim = compute_results(final_df, 'ref')
                # f1, pre, recall = eval_metrics(similarity_df, no_of_gt_tok, no_of_ex_tok)
                resultdata.append(['RefExtract', pdf.pdf_name, pdf.page_number, 'references', pre,recall,f1, lavsim])
            else:
                resultdata.append(['RefExtract', pdf.pdf_name, pdf.page_number, 'references', 0,0,0, 0])

        resultdf = pd.DataFrame(resultdata,columns=['Tool', 'ID', 'Page', 'Label', 'Precision', 'Recall', 'F1', 'SpatialDist'])
        key = dir.split('_')[1] if "_" in dir else "sample"

        filename='refextract_extract_ref_' + key + '.csv'
        outputf='./Data/results/refextract/' + filename
        os.makedirs(os.path.dirname(outputf), exist_ok=True)
        
        resultdf.to_csv(outputf, index=False)
        if refdir and os.path.exists(refdir):
            print(f"[INFO] Cleaning up {refdir}")
            shutil.rmtree(refdir)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] Exception during execution: {e}")
