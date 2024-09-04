import numpy as np
import os
import time
import openpyxl
import pandas as pd
import shutil
logis_exe_path=os.getcwd()+'/logisfr/'+'logisfr.exe '

C5_validation_1m=pd.DataFrame()
C5_validation_2m=pd.DataFrame()
C5_validation_3m=pd.DataFrame()
C5_validation_4m=pd.DataFrame()
C5_validation_5m=pd.DataFrame()
ini_path=[]
templist=[]
'''
def read_all_ini(tofsn_path):
    read_path=tofsn_path
    files=os.listdir(read_path)
    #print('in read_all_ini__________________')
    findtype='.ini'
    for file in files:
        #print(file)
        if (file.endswith(findtype) ):
            find_inipath=read_path+'\\'+file
            new_ini_path.append(find_inipath)
            #print('find ini file====================')
            #print(find_inipath)
    #print(new_ini_path)
    return new_ini_path
    #print(new_ini_path)
'''
def savedata_to_csv(folder):
    #C5_validation_1m.sort_index()
    #C5_validation_1m.reset_index(inplace=True)

    # C5_validation_1m.reset_index(inplace=True).set_index('SNname')
    #C5_validation_2m.sort_index()
    #C5_validation_2m.reset_index(inplace=True)
    #C5_validation_3m.sort_index()
    #C5_validation_3m.reset_index(inplace=True)
    #C5_validation_4m.sort_index()
    #C5_validation_4m.reset_index(inplace=True)
    #C5_validation_5m.sort_index()
    #C5_validation_5m.reset_index(inplace=True)
    to_temp_xlsx(folder)
def savecsv(path):

    C5_validation_1m.sort_index()
    C5_validation_1m.reset_index()

    #C5_validation_1m.reset_index(inplace=True).set_index('SNname')
    C5_validation_2m.sort_index()
    C5_validation_2m.reset_index()
    C5_validation_3m.sort_index()
    C5_validation_3m.reset_index()
    C5_validation_4m.sort_index()
    C5_validation_4m.reset_index()
    C5_validation_5m.sort_index()
    C5_validation_5m.reset_index()
    #print(C5_validation_1m.columns)
    to_xlsx(path)
def to_temp_xlsx(path):
    global templist
    templist.append(path + '\\' + 'temp.xlsx')
    writer=pd.ExcelWriter(path+'\\'+'temp.xlsx',engine='openpyxl')
    C5_validation_1m.to_excel(writer,sheet_name='1m_result',index=True)
    C5_validation_2m.to_excel(writer, sheet_name='2m_result', index=True)
    C5_validation_3m.to_excel(writer, sheet_name='3m_result', index=True)
    C5_validation_4m.to_excel(writer, sheet_name='4m_result', index=True)
    C5_validation_5m.to_excel(writer, sheet_name='5m_result', index=True)

    #dic.to_excel(writer,sheet_name='txtlist',index=False)
    #resultdic.to_excel(writer,sheet_name='Check_value_result',index=False)
    writer.close()
def to_xlsx(path):
    writer=pd.ExcelWriter(path+'\\'+'C5_RGB_anlysis_temp.xlsx',engine='openpyxl')
    dic={'Unnamed: 0':'SNname'}
    C5_validation_1m.rename(columns=dic,inplace=True)
    C5_validation_2m.rename(columns=dic,inplace=True)
    C5_validation_3m.rename(columns=dic,inplace=True)
    C5_validation_4m.rename(columns=dic,inplace=True)
    C5_validation_5m.rename(columns=dic,inplace=True)

    print(C5_validation_1m)

    C5_validation_1m.to_excel(writer,sheet_name='1m_result',index=False)
    C5_validation_2m.to_excel(writer, sheet_name='2m_result', index=False)
    C5_validation_3m.to_excel(writer, sheet_name='3m_result', index=False)
    C5_validation_4m.to_excel(writer, sheet_name='4m_result', index=False)
    C5_validation_5m.to_excel(writer, sheet_name='5m_result', index=False)

    #dic.to_excel(writer,sheet_name='txtlist',index=False)
    #resultdic.to_excel(writer,sheet_name='Check_value_result',index=False)
    writer.close()
def loadcsv_to_data(folder):
    global C5_validation_1m,C5_validation_2m,C5_validation_3m,C5_validation_4m,C5_validation_5m
    #data = pd.read_excel(folder+'\\'+'temp.xls')

    #excelname = folder + '\\' + 'temp.xlsx'
    for excelname in templist:
        xls = pd.ExcelFile(excelname)
        sheetname = xls.sheet_names
        for sheet in sheetname:
            data = pd.read_excel(excelname, sheet_name=sheet)
            if sheet=='1m_result':
                C5_validation_1m=C5_validation_1m._append(data,ignore_index=True)

            if sheet=='2m_result':
                C5_validation_2m = C5_validation_2m._append(data,ignore_index=True)

            if sheet=='3m_result':
                C5_validation_3m = C5_validation_3m._append(data,ignore_index=True)

            if sheet=='4m_result':
                C5_validation_4m = C5_validation_4m._append(data,ignore_index=True)
            if sheet=='5m_result':
                C5_validation_5m = C5_validation_5m._append(data,ignore_index=True)

def dataAnlysis(dis,SNname,imagename,value):
    global C5_validation_1m,C5_validation_2m,C5_validation_3m,C5_validation_4m,C5_validation_5m
    if dis=='1m':
        #C5_validation_1m.loc['SNname',0]=SNname
        C5_validation_1m.loc[SNname,imagename]=value
    if dis=='2m':

        C5_validation_2m.loc[SNname,imagename]=value
    if dis=='3m':

        C5_validation_3m.loc[SNname,imagename]=value
    if dis=='4m':

        C5_validation_4m.loc[SNname,imagename]=value
    if dis=='5m':

        C5_validation_5m.loc[SNname,imagename]=value



    #print(imagename)
    #print(C5_validation_1m)
def outputini(outputpath,imname,wordtemp):

    with open(outputpath, 'r') as f1:
        lines = f1.readlines()
        word = lines

        wordtemp = wordtemp + imname + '\n'
        wordtemp = wordtemp + '\n' + lines[1] + '\n'

        sfr = lines[1].split(',')[0]
        sfrnumber = sfr.split(':')[1].strip()
    return wordtemp,sfrnumber

def cleansettings():
    global ini_path,wordtemp,word
    wordtemp = ''
    word = ''
    ini_path=[]
def cleansettings2():
    global ini_path, wordtemp, word,C5_validation_1m,C5_validation_2m,C5_validation_3m,C5_validation_4m,C5_validation_5m
    wordtemp = ''
    word = ''
    ini_path = []
    C5_validation_1m = pd.DataFrame()
    C5_validation_2m = pd.DataFrame()
    C5_validation_3m = pd.DataFrame()
    C5_validation_4m = pd.DataFrame()
    C5_validation_5m = pd.DataFrame()
def runlogis(folder):
    global ini_path
    wordtemp = ''
    word = ''
    retry_cnts = 5
    #print(ini_path)
    #ini_path = read_all_ini(folder)
    number_ini = len(ini_path)
    # print(number_ini)
    start = 1
    dis = ['1m', '2m', '3m', '4m', '5m']
    # dis = ['1m']
    # current_path = os.getcwd()
    # print("current path= "+current_path)
    for distance in dis:
        print('current distance==========='+distance+'================')
        wordtemp = wordtemp + '============' + distance + '============' + '\n'
        for i in range(0, number_ini):
            if distance in ini_path[i]:

                print('current ini= ' + ini_path[i])
                # os.chdir(tofsn_path)
                #os.system('logisfr.exe ' + '--config ' + ini_path[i])
                os.system(logis_exe_path + '--config ' + ini_path[i])


                # imname = ini_path[i].split("/")[-1]
                imname = ini_path[i].split(".")[0]
                imname = imname.split("\\")[-1]
                imname = imname.strip("\/")
                #print('imname= ' + imname)

                outputpath = folder + '\\output\\' + 'logisfr.output.ini'
                imgpath = folder + '\\output\\' + 'imgdst.jpg'
                modoutputimgpath = folder + '\\output\\' + imname + '_result.jpg'

                # print('log_path= '+path)
                for retry in range(retry_cnts):
                    if (os.path.isfile(outputpath)):
                        ##print('no output ini')
                        wordtemp,sfrnumber=outputini(outputpath, imname,wordtemp)
                        break
                    else:
                        print('no output ini please wait' + str(retry))
                        time.sleep(1)
                for retry in range(retry_cnts):
                    if (os.path.isfile(imgpath)):
                        shutil.copyfile(imgpath, modoutputimgpath)
                        break

                    else:
                        print('no output image please wait' + str(retry))
                        time.sleep(1)
                dataAnlysis(distance, folder, imname,sfrnumber)




                #with open(outputpath, 'r') as f1:
                #    lines = f1.readlines()
                #    word = lines

                #    wordtemp = wordtemp + imname + '\n'
                #    wordtemp = wordtemp + '\n' + lines[1] + '\n'




    new_path = folder + '//output//' + 'Output_total_sfr.ini'
    with open(new_path, 'w') as f2:
        f2.writelines(wordtemp)
        # f2.writelines('/n')
    print('Folder Done!')
if __name__ == '__main__':

    tofsn_path = sys.argv[1]
    #tofsn_path = r'C:\Users\jason\Desktop\0809_C3C4\C5\CO1-17ABOK-00078_3\CO1-17ABOK-00078'
    #tofsn_path=r'C:\Users\jason\Desktop\0809_C3C4\C5'
    #tofsn_path = r'C:\Users\jason\Desktop\0809_C3C4\C5\CO1-17ABOK-00078_2\CO1-17ABOK-00078'
    #tofsn_path = r'C:\Users\jason\Desktop\0809_C3C4\C5\CO1-17ABOK-00078_1'

    path = tofsn_path + '\\'
    for folder, subfolder, files in os.walk(path):
        #print('current folder ===>'+folder)


        csvtype='.xlsx'
        findtype = '.ini'
        hascsv=0
        hasini=0
        #hasini = 0
        # if(files):
        #for name in files:
        #    if name.endswith(findtype) and 'output' not in folder:
        #        hasini = 1
            # print(files)
        if 'output' not in folder and 'tmp' not in folder and '_internal' not in folder:
            for name in files:
                if name.endswith(csvtype) and name=='temp.xlsx':
                    hascsv=1
                    #csvname=folder+'\\'+name
                if name.endswith(findtype):
                    print('current folder ===>' + folder)
                    find_inipath = folder + '\\' + name
                    ini_path.append(find_inipath)
                    hasini=1
        if hascsv==1:
            #loadcsv_to_data(folder)
            templist.append(folder+'\\'+'temp.xlsx')
            print("exist temp.xslx")
            print("load temp.xlsx and skip folder")
            continue
        if 'output' not in folder and 'tmp' not in folder and hasini==1 and hascsv==0:
            print("runfolder")
            runlogis(folder)
            savedata_to_csv(folder)
        #cleansettings()
        cleansettings2()
    loadcsv_to_data(folder)
    savecsv(path)
    #print(C5_validation_1m.columns)
    print("end of program")
    '''
            wordtemp = ''
            word=''
            retry_cnts = 10
            print(files)
            ini_path=read_all_ini(folder)
            number_ini =len(ini_path)
            #print(number_ini)
            start=1
            dis=['1m','2m','3m','4m','5m']
        #dis = ['1m']
        #current_path = os.getcwd()
        #print("current path= "+current_path)
            for distance in dis:
                wordtemp=wordtemp+'============'+distance+'============'+'\n'
                for i in range(0, number_ini):
                    if distance in ini_path[i]:




                        print('current ini= '+ini_path[i])
                    #os.chdir(tofsn_path)
                        os.system('logisfr.exe ' + '--config ' + ini_path[i])

                    #imname = ini_path[i].split("/")[-1]
                        imname = ini_path[i].split(".")[0]
                        imname = imname.split("\\")[-1]
                        imname = imname.strip("\/")
                        print('imname= '+imname)

                        outputpath = folder+'\\output\\' + 'logisfr.output.ini'
                        imgpath=folder+'\\output\\'+'imgdst.jpg'
                        modoutputimgpath=folder+'\\output\\'+imname+'_result.jpg'


                    #print('log_path= '+path)
                        for retry in range(retry_cnts):
                            if(os.path.isfile(outputpath)):
                            ##print('no output ini')
                                break
                            else:
                                print('no output ini please wait'+str(retry) )
                                time.sleep(1)
                        for retry in range(retry_cnts):
                            if(os.path.isfile(imgpath)):
                                shutil.copyfile(imgpath,modoutputimgpath)

                            else:
                                print('no output image please wait'+str(retry) )
                                time.sleep(1)
            #print(retry)
        #print('after retry')
                        with open(outputpath, 'r') as f1:
                            lines = f1.readlines()
                            word=lines

                            wordtemp = wordtemp + imname+'\n'
                            wordtemp = wordtemp + '\n' + lines[1]+'\n'
            #print(wordtemp)


                #with open(path, 'w') as f2:
            #lines = f2.readlines()
                    #f2.writelines(word)

                    cleansettings()

                new_path = folder+'//output//' + 'Output_total_sfr.ini'
                with open(new_path, 'w') as f2:
                    f2.writelines(wordtemp)
        #f2.writelines('/n')
                print('Done!')
    '''
