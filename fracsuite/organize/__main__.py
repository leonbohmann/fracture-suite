import typer
import xlsxwriter
import os

app = typer.Typer()


@app.command()
def summarize(path: str):
    """Display a table that summarizes the contents of a database folder.

    Args:
        path (str): Path of the DB.
    """
    workbook = xlsxwriter.Workbook(os.path.join(path,'summary.xlsx'))
 
    # The workbook object is then used to add new
    # worksheet via the add_worksheet() method.
    worksheet = workbook.add_worksheet()
    
    
    worksheet.write(0, 0, "Boundary: A (allseitig), Z (zweiseitig), B (gebettet)")
    worksheet.write(1, 0, "Comment: B (Bohrung)")
    
    start_row = 10
    
    worksheet.write(start_row, 0, "Name")
    worksheet.write(start_row, 1, "Thickness")
    worksheet.write(start_row, 2, "Pre-Stress")
    worksheet.write(start_row, 3, "Boundary")
    worksheet.write(start_row, 4, "Nbr")    
    worksheet.write(start_row, 5, "Comment")
    
    row = start_row + 1
    for folder in os.listdir(path):
        if folder.count('.') < 3:
            continue
        folder_path = os.path.join(path, folder)
        
        # extract data
        sp = folder.split('.')
        thick = sp[0]
        sigma = sp[1]
        bound = sp[2]
        lsp = sp[3].split('-')
        lfnr = lsp[0]
        comment = lsp[1] if len(lsp) > 1 else ""
        
        worksheet.write(row, 0, folder)    
        worksheet.write(row, 1, thick)    
        worksheet.write(row, 2, sigma)    
        worksheet.write(row, 3, bound)    
        worksheet.write(row, 4, lfnr)    
        worksheet.write(row, 5, comment)    
        
        
        
        
        row += 1
    
    # Finally, close the Excel file
    # via the close() method.
    workbook.close()

app()