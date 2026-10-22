import os
from openpyxl import Workbook, load_workbook


def save_to_excel(data_dict, user_fields, file_path="output.xlsx"):
    if os.path.exists(file_path):
        # loading existing workbook
        wb = load_workbook(file_path)
        ws = wb.active
    else:
        wb = Workbook()
        ws = wb.active
        ws.append(user_fields)  # header

    row = ws.max_row + 1
    for col, field in enumerate(user_fields, start=1):
        ws.cell(row=row, column=col, value=data_dict[field])

    wb.save(file_path)
    