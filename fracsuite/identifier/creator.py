import os
import shutil
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas
from reportlab.pdfbase.ttfonts import TTFont
from tqdm import tqdm
from pylibdmtx.pylibdmtx import encode, decode
from PIL import Image

__path__ = os.path.dirname(__file__)

# dimensions of A4 paper in mm
A4_WIDTH = 210.0
A4_HEIGHT = 297.0

# dimensions of cell in mm
CELL_WIDTH = 70.0
CELL_HEIGHT = 37.0

# number of cells in each row and column
ROWS = int(A4_HEIGHT/CELL_HEIGHT)  # // is floor division
COLUMNS = int(A4_WIDTH/CELL_WIDTH)

CANVAS_FONT = "F25_Bank_Printer_Bold"
# CANVAS_FONT = "Consolas"
CANVAS_FONT_SIZE = 24

BARCODE_HEIGHT_FAC = 0.6
LABEL_HEIGHT_FAC = 1 - BARCODE_HEIGHT_FAC


def create_datamatrix_code(code: str, label: str) -> str:
    encoded = encode(code.encode('utf-8'), size='RectAuto')
    img = Image.frombytes('RGB', (encoded.width, encoded.height), encoded.pixels)
    os.makedirs('.out', exist_ok=True)
    name = f'.out/{label}.png'
    img.save(name)

    content = decode(img, max_count=1)
    if content[0].data.decode('utf-8') != code:
        print(f"Decoded code {content[0].data.decode('utf-8')} does not match original code {code}!")

    return os.path.abspath(name)

def create_cell(label: str, code: str, x: float, y: float, canvas: canvas.Canvas):

    # save barcode image and get its path
    fullname = CREATE_CODE(code, label)

    canvas.setFont(CANVAS_FONT, CANVAS_FONT_SIZE)
    # draw label and barcode on canvas at provided coordinates
    canvas.drawImage(fullname, x*mm , (y + 0.95 * CELL_HEIGHT * LABEL_HEIGHT_FAC)*mm, width = CELL_WIDTH*mm, height = CELL_HEIGHT*mm * BARCODE_HEIGHT_FAC, anchor='n', preserveAspectRatio=ASPECT_RATIO)
    canvas.drawCentredString((x + CELL_WIDTH / 2)*mm, (y + CELL_HEIGHT * LABEL_HEIGHT_FAC / 2)*mm , label, )
    # canvas.rect(x*mm , y*mm, CELL_WIDTH * mm, CELL_HEIGHT * mm, stroke = 1, fill = 0)


def generate_pdf(labels_codes: list, filename: str):

    fontpath= os.path.join(__path__, f"{CANVAS_FONT}.ttf")
    pdfmetrics.registerFont(TTFont(CANVAS_FONT,fontpath))
    pdfmetrics.registerFontFamily(CANVAS_FONT)

    c = canvas.Canvas(filename, pagesize=(210*mm, 297*mm))

    # coordinates of top left corner of each cell
    coordinates = [(x*CELL_WIDTH, (ROWS-1-y)*CELL_HEIGHT+0.5) for y in range(ROWS) for x in range(COLUMNS)]

    # create all cells
    for i, (label, code) in enumerate(tqdm(labels_codes)):
        # start a new page after every 24 cells
        if i > 0 and i % 24 == 0:
            c.showPage()

        x, y = coordinates[i % 24]
        create_cell(label, code, x, y, c)

    # save the pdf
    c.save()


if __name__ == "__main__":
    CREATE_CODE = create_datamatrix_code

    # stretch barcode,datamatrix not stretched!
    ASPECT_RATIO = True


    existing_label_path = os.path.join(__path__, "existing_labels.txt")
    # read existing labels from "existing_labels.txt" file
    with open(existing_label_path, "r") as f:
        existentLabels = [line.strip() for line in f.readlines()]


    thick = [4,8]
    sig = {4: [120], 8: [100]}
    bound = ["Z", "A", "B"]
    lfnr = list(range(1,6))

    labels = []

    for t in thick:
        for s in sig[t]:
            for b in bound:
                for i in lfnr:
                    label = f'{t}.{s}.{b}.{i:02d}'
                    labels.append((label,label))
    labels = [(label,label) for _,label in labels if label not in existentLabels]


    # manual labels are forced
    manual_labels = []

    # these are actually used
    used_labels = []
    for label in manual_labels:
        used_labels.append((label, label))
    used_labels = [(label,label) for _,label in used_labels if label not in existentLabels]

    for n in range(len(used_labels),24):
        if len(labels) > 0:
            used_labels.append(labels.pop(0))

    name = f'{used_labels[0][0]}-{used_labels[-1][0]}.pdf'

    generate_pdf(used_labels, name)

    if os.path.exists(".out"):
        shutil.rmtree(".out", ignore_errors=True)

    if os.path.exists(name):
        os.system(f"start {name}")

    # append used labels to existing_label_path file
    with open(existing_label_path, "a") as f:
        for label in used_labels:
            f.write(label[0] + "\n")
