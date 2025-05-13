import multiprocessing
import argparse
import pdfplumber
import os
from tqdm import tqdm
from pdfminer.layout import LTChar, LTLine
#from pdfminer.layout import LAParams
from pdfplumber.pdf import LAParams
import re
from collections import Counter
import pdf2image
import numpy as np
from PIL import Image



def within_bbox(bbox_bound, bbox_in):
    assert bbox_bound[0] <= bbox_bound[2]
    assert bbox_bound[1] <= bbox_bound[3]
    assert bbox_in[0] <= bbox_in[2]
    assert bbox_in[1] <= bbox_in[3]

    x_left = max(bbox_bound[0], bbox_in[0])
    y_top = max(bbox_bound[1], bbox_in[1])
    x_right = min(bbox_bound[2], bbox_in[2])
    y_bottom = min(bbox_bound[3], bbox_in[3])

    if x_right < x_left or y_bottom < y_top:
        return False

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbox_in_area = (bbox_in[2] - bbox_in[0]) * (bbox_in[3] - bbox_in[1])

    if bbox_in_area == 0:
        return False

    iou = intersection_area / float(bbox_in_area)

    return iou > 0.95

def cmyk_to_rgb(self, cmyk):
        r = 255*(1.0-(cmyk[0]+cmyk[3]))
        g = 255*(1.0-(cmyk[1]+cmyk[3]))
        b = 255*(1.0-(cmyk[2]+cmyk[3]))
        return r,g,b

def worker(pdf_file, data_dir, output_dir):
    try:
        pdf_images = pdf2image.convert_from_path(os.path.join(data_dir, pdf_file))
    except Exception as e:
        print(f"Error converting {pdf_file} to images: {e}")
        return

    page_tokens = []
    try:
        pdf = pdfplumber.open(os.path.join(data_dir, pdf_file))

    except:
        return

    
    for page_id in tqdm(range(len(pdf.pages))):
        tokens = []

        this_page = pdf.pages[page_id]
        anno_img = np.ones([int(this_page.width), int(this_page.height)] + [3], dtype=np.uint8) * 255

        words = this_page.extract_words(x_tolerance=1.5)

        lines = []
        for obj in this_page.layout._objs:
            if not isinstance(obj, LTLine):
                continue
            lines.append(obj)

        for word in words:
            word_bbox = (float(word['x0']), float(word['top']), float(word['x1']), float(word['bottom']))
            objs = []
            for obj in this_page.layout._objs:
                if not isinstance(obj, LTChar):
                    continue
                obj_bbox = (obj.bbox[0], float(this_page.height) - obj.bbox[3],
                            obj.bbox[2], float(this_page.height) - obj.bbox[1])
                if within_bbox(word_bbox, obj_bbox):
                    objs.append(obj)
            fontname = []


            for obj in objs:
                fontname.append(obj.fontname)
            if len(fontname) != 0:
                c = Counter(fontname)
                fontname, _ = c.most_common(1)[0]
            else:
                fontname = 'default'

            # format word_bbox
            width = int(this_page.width)
            height = int(this_page.height)
            f_x0 = min(1000, max(0, int(word_bbox[0] / width * 1000)))
            f_y0 = min(1000, max(0, int(word_bbox[1] / height * 1000)))
            f_x1 = min(1000, max(0, int(word_bbox[2] / width * 1000)))
            f_y1 = min(1000, max(0, int(word_bbox[3] / height * 1000)))
            word_bbox = tuple([f_x0, f_y0, f_x1, f_y1])

            # plot color annotations
            wordcolor=obj.graphicstate.ncolor
            if isinstance(wordcolor,int) and wordcolor==0:
               rgb=(0,0,0)
            elif isinstance(wordcolor,tuple) and len(wordcolor) == 3:
             rgb = (
                    int(wordcolor[0]*255),
                    int(wordcolor[1] * 255),
                    int(wordcolor[2] * 255)
                )
            # plot annotation
            x0, y0, x1, y1 = word_bbox
            x0, y0, x1, y1 = int(x0 * width / 1000), int(y0 * height / 1000), int(x1 * width / 1000), int(
                y1 * height / 1000)
            anno_color = [0, 0, 0]
            for x in range(x0, x1):
                for y in range(y0, y1):
                    anno_img[x, y] = anno_color

            # Only plain BBox print
            xx0,yy0,xx1,yy1=obj.bbox[0],obj.bbox[1],obj.bbox[2],obj.bbox[3]
            plain_bbox = tuple([xx0,yy0,xx1,yy1])

            word_bbox = tuple([str(t) for t in word_bbox])
            rgb = tuple([str(t) for t in rgb])
            word_text = re.sub(r"\s+", "", word['text'])
            plain_bbox = tuple([str(t) for t in plain_bbox])
            tokens.append((word_text,) + word_bbox + rgb + (fontname,) + plain_bbox)

        for figure in this_page.images:
            figure_bbox = (float(figure['x0']), float(figure['top']), float(figure['x1']), float(figure['bottom']))

            # format word_bbox
            width = int(this_page.width)
            height = int(this_page.height)
            f_x0 = min(1000, max(0, int(figure_bbox[0] / width * 1000)))
            f_y0 = min(1000, max(0, int(figure_bbox[1] / height * 1000)))
            f_x1 = min(1000, max(0, int(figure_bbox[2] / width * 1000)))
            f_y1 = min(1000, max(0, int(figure_bbox[3] / height * 1000)))
            figure_bbox = tuple([f_x0, f_y0, f_x1, f_y1])

            # plot annotation
            x0, y0, x1, y1 = figure_bbox
            x0, y0, x1, y1 = int(x0 * width / 1000), int(y0 * height / 1000), int(x1 * width / 1000), int(
                y1 * height / 1000)
            anno_color = [0, 0, 0]
            for x in range(x0, x1):
                for y in range(y0, y1):
                    anno_img[x, y] = anno_color
            rgb = (0,0,0)
            figure_bbox = tuple([str(t) for t in figure_bbox])
            rgb = tuple([str(t) for t in rgb])
            word_text = '##LTFigure##'
            fontname = 'default'
            tokens.append((word_text,) + figure_bbox + rgb + (fontname,))

        for line in this_page.lines:
            line_bbox = (float(line['x0']), float(line['top']), float(line['x1']), float(line['bottom']))
            # format word_bbox
            width = int(this_page.width)
            height = int(this_page.height)
            f_x0 = min(1000, max(0, int(line_bbox[0] / width * 1000)))
            f_y0 = min(1000, max(0, int(line_bbox[1] / height * 1000)))
            f_x1 = min(1000, max(0, int(line_bbox[2] / width * 1000)))
            f_y1 = min(1000, max(0, int(line_bbox[3] / height * 1000)))
            line_bbox = tuple([f_x0, f_y0, f_x1, f_y1])

            # plot annotation
            x0, y0, x1, y1 = line_bbox
            x0, y0, x1, y1 = int(x0 * width / 1000), int(y0 * height / 1000), int(x1 * width / 1000), int(
                y1 * height / 1000)
            anno_color = [0, 0, 0]
            for x in range(x0, x1 + 1):
                for y in range(y0, y1 + 1):
                    anno_img[x, y] = anno_color


            # plot color annotations
            linecolor = line["non_stroking_color"]
            if isinstance(linecolor, int) and linecolor == 0:
                rgb = (0, 0, 0)
            elif isinstance(linecolor, tuple) and len(linecolor) == 3:
                rgb = (
                    int(linecolor[0] * 255),
                    int(linecolor[1] * 255),
                    int(linecolor[2] * 255)
                )

            line_bbox = tuple([str(t) for t in line_bbox])
            rgb = tuple([str(t) for t in rgb])
            word_text = '##LTLine##'
            fontname = 'default'
            tokens.append((word_text,) + line_bbox + rgb + (fontname,))


        anno_img = np.swapaxes(anno_img, 0, 1)
        anno_img = Image.fromarray(anno_img, mode='RGB')
        page_tokens.append((page_id, tokens, anno_img))

        pdf_images[page_id].save(
            os.path.join(output_dir, pdf_file.replace('.pdf', '') + '_{}_ori.jpg'.format(str(page_id))))
        anno_img.save(
            os.path.join(output_dir, pdf_file.replace('.pdf', '') + '_{}_ann.jpg'.format(str(page_id))))
        with open(os.path.join(output_dir, pdf_file.replace('.pdf', '') + '_{}.txt'.format(str(page_id))),
                  'w',
                  encoding='utf8') as fp:
            print(f"Writing tokens for page {page_id} to file: {os.path.join(output_dir, pdf_file.replace('.pdf', '') + '_{}.txt'.format(str(page_id)))}")
            for token in tokens:
                fp.write('\t'.join(token) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the pdf files.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the output data will be written.",
    )
    args = parser.parse_args()

    pdf_files = list(os.listdir(args.data_dir))
    pdf_files = [t for t in pdf_files if t.endswith('.pdf')]
    #print(args.data_dir)

    # pool = multiprocessing.Pool(processes=1)
    for pdf_file in tqdm(pdf_files):
        # pool.apply_async(worker, (pdf_file, args.data_dir, args.output_dir))
        #print(pdf_file)
        worker(pdf_file, args.data_dir, args.output_dir)

    # pool.close()
    # pool.join()
