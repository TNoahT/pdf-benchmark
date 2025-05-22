import pathlib
from science_parse_api.api import parse_pdf

pdf_path = pathlib.Path('./Data/pdf/247.tar_1710.11035.gz_MTforGSW_black.pdf')

# Needs to have the docker image running
# docker run -p 127.0.0.1:8080:8080 --rm --init ucrel/ucrel-science-parse:3.0.1
output_dict = parse_pdf("http://127.0.0.1", pdf_path, port="8080")

print(output_dict)
