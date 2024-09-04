import pandas as pd
import csv
import numpy as np
import os

txt_files=[]
dic=pd.DataFrame()
resultdic=pd.DataFrame({'SN':[],'SameValue?':[]})
def readcsv(csv_name):
    roi_csv = pd.read_csv(csv_name, delimiter=';')
    roi_csv = roi_csv.rename(columns={"Generation Time": "Time"})
    print(roi_csv.columns)
    roi_csv = roi_csv.dropna(axis=0, how='any')
    roi_csv=roi_csv.reset_index()
    print(roi_csv['Match X/Y'])
    a = roi_csv.loc[0, 'Match X/Y']
    b = a.split(",")
    for i in range(0, 2):
        b[i] = int(float(b[i].strip('[]')))
    print('x=')
    print(b[0])
    print('ý=')
    print(b[1])
    count = 0
def subtract():
    word=roi_csv['Match X/Y'].tolist()
    b = word.split(",")
    for i in range(0, 2):
        b[i] = int(float(b[i].strip('[]')))
    return abs(b[0]-b[1])
# Press the green button in the gutter to run the script.
#pass all txt in subfolder
def parse_txt_in_path2(path):
    global txt_files
    findtype = '.txt'
    path = path + '\\'
    for folder, subfolder, files in os.walk(path):
        if '_internal' in folder:
            continue
        for file in files:
        # print(file)
            if (file.endswith(findtype) and 'Log' in file):
                find_txtpath = folder +'\\'+ file
                txt_files.append(find_txtpath)
    print(txt_files)
def parse_txt_in_path(path):
    global txt_files
    findtype='.txt'
    path = path + '\\'
    files = os.listdir(path)
    for file in files:
        # print(file)
        if (file.endswith(findtype)):
            find_txtpath = path +  file
            txt_files.append(find_txtpath)
def compareword():
    global dic,resultdic
    dic.sort_values(by=['Logname'])
    datasize = dic.shape[0]

    tempvalue=pd.DataFrame()
    Booltemp=True
    for i in range(0, datasize):

        Logname_in_csv = dic.loc[i, 'Logname']
        SN_name = Logname_in_csv.split("_")[0]
        #print("================================")
        #print(i)
        #print(SN_name)

        if i==0:
            tempvalue=dic.loc[i, :]
        temp_SN_name = tempvalue['Logname'].split("_")[0]
        if SN_name ==temp_SN_name:

            #print('SN_name ==temp_SN_name')
            #print(i)
            #print(type(dic.iloc[i, 1:]))
            samevalue=dic.iloc[i, 1:].equals(tempvalue.iloc[1:])
            Booltemp=Booltemp & samevalue
            #print(Booltemp)
        else:
            #print('SN_name !=temp_SN_name')
            #print(i)

            #print(Booltemp)
            if Booltemp==True:
                Booltemp='YES'
            else:
                Booltemp='NO'
            resultdic=resultdic._append({'SN':temp_SN_name,'SameValue?':Booltemp}, ignore_index=True)

            Booltemp=True
            tempvalue = dic.loc[i, :]
    temp_SN_name = tempvalue['Logname'].split("_")[0]
    if Booltemp == True:
        Booltemp = 'YES'
    else:
        Booltemp = 'NO'
    resultdic = resultdic._append({'SN': temp_SN_name, 'SameValue?': Booltemp}, ignore_index=True)
def to_xlsx(txt_path):
    writer=pd.ExcelWriter(txt_path+'\\'+'C2log_anlysis.xlsx',engine='openpyxl')
    dic.to_excel(writer,sheet_name='txtlist',index=False)
    resultdic.to_excel(writer,sheet_name='Check_value_result',index=False)
    writer.close()
def loadtxt_word_to_dataframe(keyword):
    global dic
    #for word in keyword:
    #    dic[word]=None
    dic=pd.DataFrame(columns=keyword)
    dic.insert(0,"Logname",[],True)
    #print(dic)

    for files in txt_files:
        print('current txt handle=====> '+files)
        filename=files.split(".")[0]
        logname=filename.split("\\")[-1]
        #SNname=filename.split("_")[0]
        #print(filename)
        #print(logname)

        val=[]
        val.append(logname)
        with open(files, encoding="utf-8") as f:
            for line in f:
                for word in keyword:
                    if word in line:
                        line_cut=line.strip().split(':')
                        #col.append(line_cut[0].strip())
                        val.append(line_cut[1].strip())

                        #dic = dic._append({col: val},ignore_index=True)
            dic = dic._append(pd.DataFrame([val],columns=dic.columns), ignore_index=True)
    #print(dic)
                        #print(line.strip())
if __name__ == '__main__':
    #txt_name=r'C:\Users\jason\Desktop\0809_C3C4\C2\CO1-17ABOK-00068_C2-TOFCV_20240802_101309.Log_OK.txt'

    #txt_path=r'C:\Users\jason\Desktop\0809_C3C4\C2\data'
    txt_path=os.getcwd()
    keyword=['UpLimit','LowLimit','VCResult']
    #parse_txt_in_path2(txt_path)#parsing all subfolder
    parse_txt_in_path(txt_path)#parsing current folder
    if txt_files == []:
        print("no txt files in all subfolder")
        print("exit exe")
        os._exit(0)
    #parse_txt_in_path(txt_path)
    loadtxt_word_to_dataframe(keyword)
    compareword()
    print(resultdic)
    to_xlsx(txt_path)

    #dic.to_csv('test.csv', encoding='utf-8', index=False)


    #readcsv(r"C:\Users\jason\Desktop\0809_C3C4\C3-DOFIE_1m\Output_202408091m.csv")






    #roi_csv = pd.read_csv(r"C:\Users\jason\Desktop\0809_C3C4\C3-DOFIE_1m\Output_202408091m.csv", delimiter= ';')
    #roi_csv = pd.read_csv(r"C:\Users\jason\Desktop\0809_C3C4\C3-DOFIE_2m\Output_20240809_2m.csv", delimiter=';')
    #print(roi_csv.columns)
    #roi_csv=roi_csv.rename(columns={"Generation Time":"Time"})
    #print(roi_csv.columns)
    #roi_csv.columns = roi_csv.columns.str.strip(',')
    #print(roi_csv.columns)
    #roi_csv = roi_csv.dropna(axis=0, how='any')
    #roi_csv.assign(sub=roi_csv.apply(subtract))
    #print(roi_csv)
    #print(roi_csv['ROI Coordinate'])
    #print(roi_csv['Match X/Y'])
    #with open(r"C:\Users\jason\Desktop\0809_C3C4\C3-DOFIE_1m\Output_202408091m.csv") as csv_read_file:
    #    csv_read_data = csv.reader(csv_read_file)
    #    csv_read_data = csv.reader(csv_read_file, delimiter=';', quotechar='|')
    #    for row in csv_read_data:
    #       print(row)



        #csv_read_list = list(csv_read_data)
        #for row in csv_read_list:
        #    print(row)
    #a = roi_csv.loc[0, 'Match X/Y']
    #b = a.split(",")
    #for i in range(0,2):
    #    b[i] = int(float(b[i].strip('[]')))
    #print('x=')
    #print(b[0])
    #print('ý=')
    #print(b[1])
    count=0
    #for i in b:

    #    i=int(float( i.strip('[]')))
        #print(i)
    #    if(count==0):
            #print('x= '+str(i))
    #    if (count == 1):
            #print('y= ' + str(i))
    #    count=count+1
    #a=roi_csv.loc[0,'ROI Coordinate']
    #b=a.split("],")
    #for i in b:
        #print(i)
        #if '13SFR' in i:
            #print(i)
        #    k=i.split(",")
        #    kk=k[0].split('[')[-1]
            #print(kk)
            #print(k[-4])
    #print(roi_csv)
    #print(b)
    #print(roi_csv.iloc[2,:])

