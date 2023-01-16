from pdf2image import convert_from_path
import os

# skript wandelt pdf-dateien im ordner "PDF_IN" zu jpg-dateien um und legt diese in "PDF_OUT" ab
# standardauflösung DPI~200

input_dir = "PDF_In"
output_dir = "PDF_Out"

files = os.listdir(input_dir)

for file in files:
    pdf = convert_from_path(os.path.join(input_dir, file))
    count = 0
    for page in pdf:
        path = os.path.join(output_dir, file)
        page.save(path + "-" + str(count) + ".jpg", "JPEG")
        count += 1
