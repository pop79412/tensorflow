
import pandas as pd
import os
import sys
import threading

#folder=r'C:\Users\jason\Desktop\0809_C3C4\C5\CO1-17ABOK-00078_1'
#name=folder+'\\'+'result.txt'
#keyword='CENTER'
#keyword2='The average value in the ROI'
#tag=0
distance=[]
sfr=[]
C5_TOF_validation_1m=pd.DataFrame()
C5_TOF_validation_2m=pd.DataFrame()
C5_TOF_validation_3m=pd.DataFrame()
C5_TOF_validation_4m=pd.DataFrame()
C5_TOF_validation_5m=pd.DataFrame()
def save_to_dataframe(folder):
    foldersplit=folder.split("\\")
    SNname=folder
    for name in foldersplit:
        if 'CO1' in name and '_' in name:
            SNname=name
    len_dis=len(distance)
    for i in range(0,len_dis):
        dis_split=distance[i].split(" ")
        dis=dis_split[0]
        dir=dis_split[1]
        val_sfr=sfr[i]
        if dis=='1m':

            C5_TOF_validation_1m.loc[SNname,dir]=val_sfr
        if dis == '2m':
            C5_TOF_validation_2m.loc[SNname, dir] = val_sfr
        if dis == '3m':
            C5_TOF_validation_3m.loc[SNname, dir] = val_sfr
        if dis == '4m':
            C5_TOF_validation_4m.loc[SNname, dir] = val_sfr
        if dis == '5m':
            C5_TOF_validation_5m.loc[SNname, dir] = val_sfr

    clear()
def opentxt(folder,name):
    global distance,sfr
    tag = 0
    #keyword = 'CENTER'
    keyword = ['CENTER', 'LEFT', 'RIGHT']
    keyword2 = 'The average value in the ROI'
    global sfr,distance
    filepath = folder + '\\' + name
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()

        for line in lines:


            if tag==1 and keyword2 in line:
                number=line.split(":")[1].strip()
                #print(line)

                #print(number)
                sfr.append(number)
                tag=0
            for word in keyword:
                if word in line:
                    distance.append(line)
                    #print(line)
                    tag = 1
            #if keyword in line:
                #print(line)
            #    distance.append(line)
            #    tag=1
    #save_to_dataframe(folder)
def clear():
    global sfr,distance
    sfr=[]
    distance=[]
def to_xlsx(path):
    #print(C5_TOF_validation_1m.index.name)
    #print(C5_TOF_validation_1m.columns)
    writer=pd.ExcelWriter(path+'\\'+'C5_TOF_anlysis_temp.xlsx',engine='openpyxl')
    C5_TOF_validation_1m.to_excel(writer,sheet_name='1m_result',index=True)
    C5_TOF_validation_2m.to_excel(writer, sheet_name='2m_result', index=True)
    C5_TOF_validation_3m.to_excel(writer, sheet_name='3m_result', index=True)
    C5_TOF_validation_4m.to_excel(writer, sheet_name='4m_result', index=True)
    C5_TOF_validation_5m.to_excel(writer, sheet_name='5m_result', index=True)

    #dic.to_excel(writer,sheet_name='txtlist',index=False)
    #resultdic.to_excel(writer,sheet_name='Check_value_result',index=False)
    writer.close()
def savecsv(path):

    C5_TOF_validation_1m.sort_index()
    C5_TOF_validation_1m.reset_index()

    C5_TOF_validation_1m.index.name='SNname'

    #C5_validation_1m.reset_index(inplace=True).set_index('SNname')
    C5_TOF_validation_2m.sort_index()
    C5_TOF_validation_2m.reset_index()
    C5_TOF_validation_2m.index.name = 'SNname'

    C5_TOF_validation_3m.sort_index()
    C5_TOF_validation_3m.reset_index()
    C5_TOF_validation_3m.index.name = 'SNname'
    C5_TOF_validation_4m.sort_index()
    C5_TOF_validation_4m.reset_index()
    C5_TOF_validation_4m.index.name = 'SNname'
    C5_TOF_validation_5m.sort_index()
    C5_TOF_validation_5m.reset_index()
    C5_TOF_validation_5m.index.name = 'SNname'
    #print(C5_validation_1m.columns)
    to_xlsx(path)
def total_analysis():
    global C5_TOF_validation_1m,C5_TOF_validation_2m,C5_TOF_validation_3m,C5_TOF_validation_4m,C5_TOF_validation_5m
    dataframelist=[C5_TOF_validation_1m]
    #dataframelist=[C5_TOF_validation_1m,C5_TOF_validation_2m,C5_TOF_validation_3m,C5_TOF_validation_4m,C5_TOF_validation_5m]
    for dataframe_name in dataframelist:
        print(dataframe_name)

def tof_eric(path):
    command="run_roi.bat "+path+" *depth.raw"
    os.system(command)
def drawcurve(path):
    excelpath=path+'\\'+'C5_TOF_anlysis_temp.xlsx'


if __name__ == '__main__':
    tofsn_path = r'C:\Users\jason\Desktop\0809_C3C4\C5\CO1-17ABOK-00078_3\CO1-17ABOK-00078'
    #tofsn_path = r'C:\Users\jason\Desktop\0809_C3C4\C5'
    path = tofsn_path + '\\'
    for folder, subfolder, files in os.walk(path):
        #print('current folder ===>'+folder)

        hasresult=0
        hasdepth=0
        if 'output' not in folder and 'tmp' not in folder and '_internal' not in folder:
            print('current folder ===>' + folder)
            for name in files:
                if name=='result.txt':
                    hasresult=1
                if 'depth.raw' in name:
                    hasdepth=1
                    #opentxt(folder,name)
                    #save_to_dataframe(folder)
            if hasresult == 0 and hasdepth==1:
                print("===TOF depth counting===")
                #print(folder.split('\\')[:-2])
                pre_folder= '\\'.join(folder.split('\\')[:-2])
                print('exe go'+pre_folder)
                t = threading.Thread(target=tof_eric(pre_folder))
                t.start()
                t.join()
                hasresult=1
            if hasresult==1 and hasdepth==1:
                opentxt(folder, 'result.txt')
                save_to_dataframe(folder)

                #csvname=folder+'\\'+name
    #print(C5_TOF_validation_1m)
    #total_analysis()
    #os.system('taskkill /f /im run_roi.bat')

    savecsv(path)
    drawcurve(path)