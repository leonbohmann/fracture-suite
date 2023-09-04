import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas
from reportlab.pdfbase.ttfonts import TTFont
from tqdm import tqdm
from pylibdmtx.pylibdmtx import encode
from PIL import Image

# dimensions of A4 paper in mm
A4_WIDTH = 210.0
A4_HEIGHT = 297.0

# dimensions of cell in mm
CELL_WIDTH = 70.0
CELL_HEIGHT = 37.0

# number of cells in each row and column
ROWS = int(A4_HEIGHT/CELL_HEIGHT)  # // is floor division
COLUMNS = int(A4_WIDTH/CELL_WIDTH)

# CANVAS_FONT = "ConsolaMono-Bold"
CANVAS_FONT = "F25_Bank_Printer_Bold"
CANVAS_FONT_SIZE = 23

BARCODE_HEIGHT_FAC = 0.6
LABEL_HEIGHT_FAC = 1 - BARCODE_HEIGHT_FAC

def create_datamatrix_code(code: str, label: str) -> str:
    encoded = encode(code.encode('utf-8'), )
    img = Image.frombytes('RGB', (encoded.width, encoded.height), encoded.pixels)
    os.makedirs('out', exist_ok=True)
    name = f'out/{label}.png'
    img.save(name)
    return os.path.abspath(name)
    
def create_cell(label: str, code: str, x: float, y: float, canvas: canvas.Canvas):

    # save barcode image and get its path
    fullname = CREATE_CODE(code, label)

    canvas.setFont(CANVAS_FONT, CANVAS_FONT_SIZE)
    # draw label and barcode on canvas at provided coordinates
    canvas.drawCentredString((x + CELL_WIDTH / 2)*mm, (y + CELL_HEIGHT * LABEL_HEIGHT_FAC / 2)*mm , label )
    canvas.drawImage(fullname, x*mm , (y + CELL_HEIGHT * LABEL_HEIGHT_FAC)*mm, width = CELL_WIDTH*mm, height = CELL_HEIGHT*mm * BARCODE_HEIGHT_FAC, anchor='n', preserveAspectRatio=ASPECT_RATIO)    
    # canvas.rect(x*mm , y*mm, CELL_WIDTH * mm, CELL_HEIGHT * mm, stroke = 1, fill = 0)
    
    
def generate_pdf(labels_codes: list, filename: str):
    __path__ = os.path.dirname(__file__)
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
    
    thick = [4,8,12]
    sig = {4: [x for x in range(70,150,10)], 8: [x for x in range(70,150,10)], 12: [x for x in range(40,120,10)]}
    bound = ["Z", "A", "B"]
    lfnr = list(range(1,11))
    
    labels = []
    
    for t in thick:
        for s in sig[t]:
            for b in bound:
                for i in lfnr:
                    label = f'{t}.{s}.{b}.{i:02d}'
                    labels.append((label,label))
                    # f"lb.de/specimen/{label}"
             
             
    manual_labels = []
    manual_labels.append("8.110.A.11")
    manual_labels.append("8.110.A.12")
    manual_labels.append("8.110.B.11")
    manual_labels.append("8.110.B.12")
    manual_labels.append("8.110.Z.13")
    manual_labels.append("8.110.Z.14")
    
    manual_labels.append("8.110.B.1")
    manual_labels.append("8.110.B.2")
    manual_labels.append("8.110.B.3")
    manual_labels.append("8.110.B.4")
    manual_labels.append("8.110.B.5")
    
    manual_labels.append("8.110.A.1")
    manual_labels.append("8.110.A.2")
    manual_labels.append("8.110.A.3")
    manual_labels.append("8.110.A.4")
    manual_labels.append("8.110.A.5")
    
    manual_labels.append("8.140.B.1")
    manual_labels.append("8.140.B.2")
    manual_labels.append("8.140.B.3")
    manual_labels.append("8.140.B.4")
    manual_labels.append("8.140.B.5")
    
    manual_labels.append("8.140.A.1")
    manual_labels.append("8.140.A.2")
    manual_labels.append("8.140.A.3")
    
    
    labels = [(label, label) for label in manual_labels]
    generate_pdf(labels, "output.pdf")