import gspread
import re
import os
import sys
from gspread_formatting import get_user_entered_format, format_cell_range

def read_experiment_data(file_path, method_name):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = {
        "Method": method_name,
        "mIoU": "",
        "mAcc": "",
        "allAcc": "",
    }

    # 뒤에서부터 검색
    for i in range(len(lines) - 1, -1, -1):
        if "Syncing" in lines[i] and i + 1 < len(lines):
            match = re.search(r"Val result: mIoU/mAcc/allAcc ([\d.]+)/([\d.]+)/([\d.]+)", lines[i + 1])
            if match:
                data["mIoU"] = match.group(1)
                data["mAcc"] = match.group(2)
                data["allAcc"] = match.group(3)
                break

    return data

def copy_format_from_previous_row(sheet, dest_row):
    source_row = dest_row - 1
    columns = [chr(i) for i in range(ord('B'), ord('Z') + 1)]

    for col in columns:
        source_cell = f'{col}{source_row}'
        dest_cell = f'{col}{dest_row}'
        fmt = get_user_entered_format(sheet, source_cell)
        if fmt:
            format_cell_range(sheet, dest_cell, fmt)

def save_gspread(result_path, method_name, flag):
    gc = gspread.service_account(filename='/workdir/gspread/account.json')
    result_data = read_experiment_data(result_path, method_name)
    #gc = gspread.service_account()
    sh = gc.open("3dgs-pc")
    sheet = sh.worksheet(flag)

    all_values = sheet.col_values(2)
    row_number = 2 if len(all_values) <= 1 else len(all_values) + 1

    copy_format_from_previous_row(sheet, row_number)
    print(f"Insert in row {row_number}: {result_data}")

    updates = [
        {'range': f'B{row_number}', 'values': [[result_data["Method"]]]},
        {'range': f'C{row_number}', 'values': [[result_data["mIoU"]]]},
        {'range': f'E{row_number}', 'values': [[result_data["mAcc"]]]},
        {'range': f'G{row_number}', 'values': [[result_data["allAcc"]]]},
    ]

    sheet.batch_update(updates)
    print("✅ Data uploaded successfully!")

if __name__ == "__main__":
    result_path = sys.argv[1]
    method_name = sys.argv[2]
    flag = sys.argv[3]
    save_gspread(result_path, method_name, flag)
