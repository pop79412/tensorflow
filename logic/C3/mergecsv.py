# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

#import os
#import sys
import pandas as pd
def mergecsv():
    c3_csv_files=["1m.csv","2m.csv","3m.csv","4m.csv","5m.csv"]
    writer=pd.ExcelWriter("c3_validation_output.xlsx",engine="xlsxwriter")
    for csv_file in c3_csv_files:
        df=pd.read_csv(csv_file)
        sheet_name=csv_file.split(".")[0]
        df.to_excel(writer,sheet_name=sheet_name,index=False)
    writer.close()
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    mergecsv()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
