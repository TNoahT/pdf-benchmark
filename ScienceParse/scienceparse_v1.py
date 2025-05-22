import os
import csv
import pathlib
from glob import glob
from os import path
import pandas as pd
from science_parse_api.api import parse_pdf
from tqdm import tqdm
from scipy.spatial.distance import cdist
from PyPDF2 import PdfReader, PdfWriter
from Levenshtein import ratio

class PDF:
    def __init__(self, page_number=None, pdf_name=None, filepath=None, txt_name=None, txt_data=None):
        self.page_number = page_number
        self.pdf_name = pdf_name
        self.filepath=filepath
        self.txt_name=txt_name
        self.txt_data=txt_data

def locate_data(dir):
    """
    Function to find Text their respective PDF files from DocBank dataset.
    :param dir: Base directory
    :return: list of text and pdf files.
    """
    pdffiles=glob(path.join(dir,"*.{}".format('pdf')))
    txtfiles=glob(path.join(dir,"*.{}".format('txt')))
    return pdffiles, txtfiles



def load_data(dir, labe):
    """
    Function creates the PDF objects for the gives base directory of DocBank dataset.
    :param dir: Base location of DocBank dataset.
    :return: List of PDF Objects.
    """
    txtf = glob(path.join(dir,"*.{}".format('txt')))
    PDFlist=[]

    for txt in txtf:
        nwe=os.path.splitext(os.path.basename(txt))[0] # 2.tar_1801.00617.gz_idempotents_arxiv_4.txt --> 2.tar_1801.00617.gz_idempotents_arxiv_4
        keyword=nwe.rpartition('_')[0] # 2.tar_1801.00617.gz_idempotents_arxiv_4 --> 2.tar_1801.00617.gz_idempotents_arxiv
        page_number=nwe.split('_')[-1]  # 2.tar_1801.00617.gz_idempotents_arxiv_4 --> 4
        pdfn = dir + "/" +keyword + "_black.pdf"
        if os.path.isfile(pdfn):
            pdf_name=os.path.basename(pdfn)
            txt_name=os.path.basename(txt)
            txtdf=pd.read_csv(txt,sep='\t',quoting=csv.QUOTE_NONE, encoding='latin1',usecols=[0,1,2,3,4,9], names=["token", "x0", "y0", "x1", "y1","label"])
            found = txtdf[txtdf['label']==labe].index.tolist()
            if len(found) != 0:
                PDFlist.append(PDF(page_number,pdf_name,dir,txt_name,txtdf))
    return PDFlist

def load_all_data(dir) :
    """
    Function creates the PDF objects for the gives base directory of DocBank dataset.
    Removes the biais created by only selecting relevant pages for a label, avoiding false positives.
    :param dir: Base location of DocBank dataset.
    :return: List of PDF Objects.
    """
    txtf = glob(path.join(dir,"*.{}".format('txt')))
    PDFlist=[]

    for txt in txtf:
        nwe=os.path.splitext(os.path.basename(txt))[0] # 2.tar_1801.00617.gz_idempotents_arxiv_4.txt --> 2.tar_1801.00617.gz_idempotents_arxiv_4
        keyword=nwe.rpartition('_')[0] # 2.tar_1801.00617.gz_idempotents_arxiv_4 --> 2.tar_1801.00617.gz_idempotents_arxiv
        page_number=nwe.split('_')[-1]  # 2.tar_1801.00617.gz_idempotents_arxiv_4 --> 4
        pdfn = dir + "/" +keyword + "_black.pdf"
        if os.path.isfile(pdfn):
            pdf_name=os.path.basename(pdfn)
            txt_name=os.path.basename(txt)
            txtdf=pd.read_csv(txt,sep='\t',quoting=csv.QUOTE_NONE, encoding='latin1',usecols=[0,1,2,3,4,9], names=["token", "x0", "y0", "x1", "y1","label"])
            PDFlist.append(PDF(page_number,pdf_name,dir,txt_name,txtdf))
    return PDFlist

def crop_pdf(pdfpath,pdfname, pagenumber):
    #pages = [pagenumber] # page 1, 3, 5
    try:
        pdf_file_name = pdfname
        file_base_name = pdf_file_name.replace('.pdf', '')
        pdf = PdfReader(pdfpath + os.sep + pdf_file_name, strict=False)
        pdfWriter = PdfWriter()
        #for page_num in pages:
        pdfWriter.add_page(pdf.pages[int(pagenumber)])
        cropped_file=pdfpath + os.sep + file_base_name + '_subset_' + pagenumber + '.pdf'
        with open(cropped_file, 'wb') as f:
            pdfWriter.write(f)
            f.close()
        return cropped_file
    except Exception as e :
        print(f"[ERROR] Failed to crop {pdfname} page {pagenumber}: {e}")
        pass

def process_tokens(data_gt, extracted_file, label):

    # DocBank has hyphanated words which are treated as seperate tokens. Affects the evaluation metrics hence fixing the hyphanated word problem in GT.
    # if len(data_gt) != 0:
    #     data_gt=fix_hyphanated_tokens(data_gt)

    if data_gt.empty:
        return pd.DataFrame(columns=[label + '_ex', label + '_gt'])

    #text=pd.read_csv(extracted_file, quoting=csv.QUOTE_NONE,encoding='latin1',engine='python', sep='\n', header=None).astype(str)[0].str.cat()
    with open(extracted_file, encoding='latin1') as f:
        lines = f.readlines()
    text = ''.join([line.strip() for line in lines])
    
    data_ex = pd.DataFrame([text], columns=['extracted'])
    # Merge all the tokens and create a stream
    data_gt = data_gt['token'].astype(str).str.cat(sep=' ')
    data_ex = data_ex['extracted'].astype(str).str.cat(sep=' ')

    ex = label + '_ex'
    gt = label + '_gt'

    d = {
        ex: [data_ex],
        gt: [data_gt],
    }
    final_df = pd.DataFrame(d)
    return final_df

def compute_metrics(pdf, extracted_file, label):
    txt_data=pdf.txt_data
    data_gt=txt_data.loc[(txt_data['label'] == label) & (txt_data['token'] != "##LTLine##")]

    if data_gt.empty:
        return 0,0,0,0

    final_df=process_tokens(data_gt,extracted_file, label)
    if len(data_gt) != 0:
        f1,pre,recall,lavsim=compute_results(final_df, label)
        return f1,pre, recall, lavsim
    
def compute_sim_matrix(ex_nump, gt_nump):
    """
    This function computes the similarity matrix for each word from a numpy array. or it can also compare whole abstract as a collated tokens.
    :param ex_nump: Extracted paragraph as numpy array
    :param gt_nump: Ground truth paragraph as numpy array
    :return: Similarity Matrix with Lavensteins similarity index.
    """
    matrix = cdist(ex_nump.reshape(-1, 1), gt_nump.reshape(-1, 1), lambda x, y: ratio(x[0], y[0]))
    df = pd.DataFrame(data=matrix, index=ex_nump, columns=gt_nump)
    return df

def compute_tpfp(matrix):
    """
    This function considers Extracted token as Ground-truth token when its Levenshteins similarity index is > 0.7. Otherwise it is non-gt token.
    :param matrix: Similarity Matrix
    :return: Number of GT in ET, Number of Non GT
    """
    tp=0
    fp=0
    rows=matrix.shape[0]
    cols=matrix.shape[1]
    for x in range(0,rows):
        for y in range(0,cols):
            if matrix.iloc[x,y] > 0.7:
                flag=True
                break
            else:
                flag=False
        if flag is True:
            tp=tp+1
        else:
            fp=fp+1
    return tp,fp

def compute_scores(tp,fp, gttoken):
    """
    Function to compute the evaluation metrics.
    :param tp: Number of GT in ET
    :param fp: Number of Non-GT in ET
    :param gttoken: Number of GT
    :return: Precision, Recall and F1 Score
    """
    prec=tp/(tp+fp)
    recall= tp/gttoken
    if prec==0 and recall==0:
        return 0,0,0
    else:
        f1_score= (2 * prec * recall)/ (prec + recall)
        return f1_score, prec, recall


def compute_results(dataf, field):
    """
    Function computes the similarity index and string distance for extracted and ground truth tokens.
    :param dataf: dataframe with one row of extracted and ground truth
    :param field: Token for which similarity is computed. e.g. Title, Abstract, Author.
    :return: Dataframe with computed similarity indices, no. of extracted tokens, no. of ground truth tokens.
    """
    extracted=field + '_ex'
    groundtruth=field + '_gt'
    df_extracted=dataf[[extracted]]
    df_groundtruth=dataf[[groundtruth]]
    #df_extracted = df_extracted.astype(str).applymap(str.split).apply(pd.Series.explode,axis=0).reset_index().drop("index", 1)
    df_extracted = (
        df_extracted.astype(str)
        .apply(lambda col: col.str.split())
        .apply(pd.Series.explode, axis=0)
        .reset_index()
        .drop("index", axis=1)
    )

    # Obsolete syntaxe
    #df_groundtruth = df_groundtruth.astype(str).applymap(str.split).apply(pd.Series.explode, axis=0).reset_index().drop("index", 1)
    df_groundtruth = (
        df_groundtruth.astype(str)
        .apply(lambda col: col.str.split())
        .apply(pd.Series.explode, axis=0)
        .reset_index()
        .drop("index", axis=1)
    )
    #df_extracted=df_extracted.applymap(str)
    df_extracted = df_extracted.apply(lambda col: col.astype(str))

    # Computing similarity not considering reading order.
    df_extractednp = dataf[extracted].to_numpy()
    df_groundtruthnp = dataf[groundtruth].to_numpy()
    matrix = compute_sim_matrix(df_extractednp, df_groundtruthnp)

    # if field == 'paragraph':
    #     return 0,0,0,matrix.iloc[0,0]

    # Computing the count of tokens before adding the NaN ! This is important to avoid false calculation of the metrics.
    no_of_gt_tokens=len(df_groundtruth.index)
    # Adding NaN to the dataframes if their length is not equal.
    if (len(df_extracted.index)-len(df_groundtruth.index)) < 0:
        diff=abs(len(df_extracted.index)-len(df_groundtruth.index))
        for i in range(diff):
            df_extracted.loc[len(df_extracted)] = 'NaN'
    if (len(df_extracted.index)-len(df_groundtruth.index)) > 0:
        diff = (len(df_extracted.index)-len(df_groundtruth.index))
        for i in range(diff):
            df_groundtruth.loc[len(df_groundtruth)] = 'NaN'
    if (len(df_extracted.index)-len(df_groundtruth.index)) == 0:
        df_e = df_extracted.to_numpy()
        df_g = df_groundtruth.to_numpy()
        simmatrix = compute_sim_matrix(df_e, df_g)

        tp,fp=compute_tpfp(simmatrix)
        f1,prec,recal=compute_scores(tp,fp,no_of_gt_tokens)

    return f1,prec,recal, matrix.iloc[0,0]

def get_refstring(reflist):
    refs=[]
    if len(reflist) == 0:
        return refs
    else:
        for ref in reflist:
            if len(ref.get('authors')) != 0:
                authors = ' '.join(ref.get('authors'))
                reststr=' '.join(list(map(str, list(ref.values())[1:])))
                refstring = authors + reststr
                refs.append(refstring)
            else:
                reststr=' '.join(list(map(str, list(ref.values())[0:])))
                refs.append(reststr)
        return refs
def get_sectstr(sectlist, label):
    if len(sectlist) == 0:
        return "NONE"
    else:
        sectstr=""
        for sect in sectlist:
            if label == 'heading':
                if 'heading' in sect:
                    sectstr = sectstr + " " + sect.get('heading')
                else:
                    continue
            if label == 'text':
                if 'text' in sect:
                    sectstr = sectstr + os.linesep + sect.get('text')
                else:
                    continue
        return sectstr


def extract_scienceparse(dir):
    host = 'http://127.0.0.1'
    port = '8080'
    label_array=['section','reference', 'paragraph', 'year']
    #label_array=['reference']
    resultdata=[]
    for label in label_array:
        PDFlist=load_data(dir, label)
        for pdf in tqdm(PDFlist):
            pdfpath=pathlib.Path(pdf.filepath + os.sep + pdf.pdf_name)
            page = int(pdf.page_number) - 1
            croppedfile=crop_pdf(pdf.filepath, pdf.pdf_name, str(page))
            if isinstance(croppedfile, type(None)):
                continue
            croppedfilepath=pathlib.Path(croppedfile)

            outputfile = pdf.filepath + os.sep + os.path.splitext(os.path.basename(pdf.pdf_name))[0] + "_extracted_" + label + ".txt"

            if label == 'title':
                output_dict = parse_pdf(host, pdfpath, port)
                if isinstance(output_dict, type(None)):
                    continue
                with open(outputfile, "w") as text_file:
                    print(output_dict.get('title'), file=text_file)
                os.remove(croppedfile)
            elif label == 'abstract':
                output_dict = parse_pdf(host, pdfpath, port)
                if isinstance(output_dict, type(None)):
                    continue
                with open(outputfile, "w") as text_file:
                    print(output_dict.get('abstractText'), file=text_file)
                os.remove(croppedfile)
            elif label == 'author':
                output_dict = parse_pdf(host, pdfpath, port)
                if isinstance(output_dict, type(None)):
                    continue
                authlist=output_dict.get('authors')
                if len(authlist) == 0:
                    break
                with open(outputfile, "w") as text_file:
                    for auth in authlist:
                        print(auth.get('name'), file=text_file)
                os.remove(croppedfile)
            elif label == 'reference':
                output_dict = parse_pdf(host, croppedfilepath, port)
                if isinstance(output_dict, type(None)):
                    continue
                reflist=output_dict.get('references')
                refstring=get_refstring(reflist)
                with open(outputfile, "w") as text_file:
                    print(refstring, file=text_file)
                os.remove(croppedfile)
            elif label == 'section':
                croppedfile = crop_pdf(pdf.filepath, pdf.pdf_name, pdf.page_number)
                croppedfilepath = pathlib.Path(croppedfile)
                output_dict = parse_pdf(host, croppedfilepath, port)
                if isinstance(output_dict, type(None)):
                    continue
                sectlist=output_dict.get('sections')
                sectstr=get_sectstr(sectlist, 'heading')
                if sectstr == "" or sectstr == " ":
                    open(outputfile, 'w').close()
                else:
                    with open(outputfile, "w") as text_file:
                        print(sectstr, file=text_file)
                os.remove(croppedfile)
            elif label == 'paragraph':
                croppedfile = crop_pdf(pdf.filepath, pdf.pdf_name, pdf.page_number)
                croppedfilepath = pathlib.Path(croppedfile)
                output_dict = parse_pdf(host, croppedfilepath, port)

                if isinstance(output_dict, type(None)):
                    continue

                sectlist=output_dict.get('sections')
                sectstr=get_sectstr(sectlist, 'text')
                if sectstr == "" or sectstr == os.linesep:
                    open(outputfile, 'w').close()
                else:
                    with open(outputfile, "w") as text_file:
                        print(sectstr, file=text_file)
                os.remove(croppedfile)
            else:
                print('cannot process the label')

            if isinstance(outputfile, type(None)):
                continue
            if os.path.getsize(outputfile) == 0 or not os.path.isfile(outputfile):
                resultdata.append(['ScienceParse', pdf.pdf_name, pdf.page_number, label, 0,0,0, 0])
                os.remove(outputfile)
            else:
                f1,pre,recall, lavsim=compute_metrics(pdf, outputfile, label)
                resultdata.append(['ScienceParse', pdf.pdf_name, pdf.page_number, label, pre,recall,f1, lavsim])
                os.remove(outputfile)

    resultdf = pd.DataFrame(resultdata,
                            columns=['Tool', 'ID', 'Page', 'Label', 'Precision', 'Recall', 'F1', 'SpatialDist'])

    return resultdf




def main():
        resultdf=extract_scienceparse("./Data/Docbank_sample")
        filename='scienceparse_extract_ref.csv'
        outputf='./Data/results/ScienceParse/' + filename
        resultdf.to_csv(outputf, index=False)

if __name__ == "__main__":
    main()