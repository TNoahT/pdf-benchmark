import camelot
import os
from glob import glob
import pandas as pd
from PyPDF2 import PdfReader
import csv
from Levenshtein import ratio
import numpy as np
from scipy.spatial.distance import cdist

PDF_DIR = "./Data/Docbank_sample/"
OUTPUT_CSV = "./Data/results/Camelot/camelot_output.csv"
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

def extract_tables_from_pdf(pdf_path, page_num):
    try:
        tables = camelot.read_pdf(pdf_path, pages=str(page_num + 1), flavor='stream')
        return [' '.join(table.df.fillna('').astype(str).values.flatten()) for table in tables]
    except Exception as e:
        print(f"[ERROR] Camelot extraction failed for {pdf_path} page {page_num}: {e}")
        return []

def load_ground_truth_text(gt_path):
    try:
        gt_df = pd.read_csv(gt_path, sep='\t', quoting=csv.QUOTE_NONE, encoding='latin1', usecols=[0, 9], names=["token", "label"])
        tokens = gt_df[gt_df["label"] == "table"]["token"].astype(str)
        return ' '.join(tokens)
    except Exception as e:
        print(f"[ERROR] Failed to load GT from {gt_path}: {e}")
        return ""

def get_pdf_pagecount(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        return len(reader.pages)
    except:
        return 0

def compute_sim_matrix(ex_nump, gt_nump):
    matrix = cdist(ex_nump.reshape(-1, 1), gt_nump.reshape(-1, 1), lambda x, y: ratio(x[0], y[0]))
    return pd.DataFrame(data=matrix, index=ex_nump, columns=gt_nump)

def compute_tpfp(matrix):
    tp, fp = 0, 0
    for row in matrix.values:
        if any(cell > 0.7 for cell in row):
            tp += 1
        else:
            fp += 1
    return tp, fp

def compute_scores(tp, fp, gttoken):
    prec = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / gttoken if gttoken > 0 else 0
    f1 = (2 * prec * recall / (prec + recall)) if prec + recall > 0 else 0
    return f1, prec, recall

def compute_metrics_all(gt_text, extracted_tables, pdf_name, page_num, tool_name="Camelot", label="table"):
    gt_tokens = gt_text.split()
    if not gt_tokens:
        return []

    rows = []
    for i, extracted in enumerate(extracted_tables, start=1):
        ex_tokens = extracted.split()

        df_gt = pd.DataFrame(gt_tokens, columns=[label + '_gt'])
        df_ex = pd.DataFrame(ex_tokens, columns=[label + '_ex'])

        # Pad shorter list with NaNs to align token count
        len_gt, len_ex = len(df_gt), len(df_ex)
        if len_ex < len_gt:
            df_ex = pd.concat([df_ex, pd.DataFrame([np.nan] * (len_gt - len_ex), columns=[label + '_ex'])], ignore_index=True)
        elif len_ex > len_gt:
            df_gt = pd.concat([df_gt, pd.DataFrame([np.nan] * (len_ex - len_gt), columns=[label + '_gt'])], ignore_index=True)

        ex_np = df_ex.to_numpy().astype(str)
        gt_np = df_gt.to_numpy().astype(str)
        sim_matrix = compute_sim_matrix(ex_np, gt_np)

        tp, fp = compute_tpfp(sim_matrix)
        f1, prec, rec = compute_scores(tp, fp, len(gt_tokens))
        first_sim = sim_matrix.iloc[0, 0] if not sim_matrix.empty else 0.0

        rows.append([
            tool_name,
            pdf_name,
            page_num,
            label,
            f"{prec:.3f}",
            f"{rec:.3f}",
            f"{f1:.3f}",
            f"{first_sim:.3f}"
        ])

    return rows

def main():
    pdf_files = glob(os.path.join(PDF_DIR, "*.pdf"))
    results = []

    for pdf_path in pdf_files:
        base = os.path.basename(pdf_path).replace(".pdf", "").removesuffix("_black")
        if base.endswith("_black"):
            base = base[:-6]
        matching_txts = sorted(glob(os.path.join(PDF_DIR, base + "_*.txt")))
        if not matching_txts:
            print(f"[SKIP] No GT TXT files found for {base}")
            continue

        page_count = get_pdf_pagecount(pdf_path)
        if page_count == 0:
            print(f"[SKIP] Could not read {base}")
            continue

        for txt_path in matching_txts:
            try:
                page_num = int(os.path.splitext(txt_path)[0].split("_")[-1])
            except:
                print(f"[SKIP] Cannot parse page number from {txt_path}")
                continue

            if page_num >= page_count:
                print(f"[SKIP] Page {page_num} out of range for {base}")
                continue

            gt_text = load_ground_truth_text(txt_path)
            if not gt_text.strip():
                continue

            extracted = extract_tables_from_pdf(pdf_path, page_num)
            if not extracted:
                continue

            per_table_rows = compute_metrics_all(gt_text, extracted, base + ".pdf", page_num)
            results.extend(per_table_rows)

            print(f"[OK] {base} page {page_num}: {len(per_table_rows)} tables processed.")

    df = pd.DataFrame(results, columns=[
        "Tool", "ID", "Page", "Label", "Precision", "Recall", "F1", "SpatialDist"
    ])
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[SAVED] Results written to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()