
import pandas as pd
import os
import time
import shutil
#import threading
#logis_exe_path=r'C:\Users\jason\Desktop\0809_C3C4\MTF_Near\1\mtf_v0.2.exe'
inilist=[]
looptest_result=pd.DataFrame()

def handleini(inipath):

    with open(inipath, 'r') as f1:
        lines = f1.readlines()
        sfr = lines[1].split(',')[0]
        sfrnumber = sfr.split(':')[1].strip().strip('{}')
        print(sfrnumber)
    return sfrnumber
def save_to_dataframe(imagename,sfrnumber,count):
    looptest_result.loc[count,imagename] = sfrnumber
def looptest(ini_path,times):
    ini_split=ini_path.split("\\")
    folder="\\".join(ini_split[:-1])
    print('folder= '+folder)
    imagename=ini_split[-1].split('.')[0]
    retry_cnts = 5
    mtfinipath=folder+'\\'+'mtf_config.ini'
    shutil.copyfile( ini_path,mtfinipath)
    logis_exe_path='mtf_v0.2.exe'
    for i in range(times):
        print('times'+': '+str(i+1))
        #os.system(logis_exe_path + '--config ' + ini_path)
        os.system(logis_exe_path)
        outputpath = folder + '\\output\\' + 'mtf_output.ini'
        imgpath = folder + '\\output\\' + 'imgdst.jpg'
        image_result= folder+'\\output\\'+imagename+'.jpg'
        for retry in range(retry_cnts):
            if (os.path.isfile(outputpath)):
                ##print('no output ini')
                sfrnumber = handleini(outputpath)
                break
            else:
                print('no output ini please wait' + str(retry))
                time.sleep(1)

        save_to_dataframe(imagename,sfrnumber,i+1)
        #for retry in range(retry_cnts):
        #    if (os.path.isfile(imgpath)):
        #        shutil.copyfile(imgpath, image_result)
        #        break

        #    else:
        #        print('no output image please wait' + str(retry))
        #        time.sleep(1)

def save_to_csv(path):
    looptest_result.to_csv(path+'\\'+'looptest_result' + '.csv', encoding='utf-8', index=True)
def run_looptest(path,times):
    for ini in inilist:
        print('current do in file: '+ini)
        looptest(ini,times)
    save_to_csv(path)



def findini(path):
    global inilist
    path = path + '\\'
    findtype='.ini'
    for folder, subfolder, files in os.walk(path):
        #print(files)
        if 'output' not in folder and '_internal' not in folder:
            for file in files :
                if findtype in file and file!='mtf_config.ini':
                    inilist.append(path+file)
    print('total ini in list')
    print(inilist)


if __name__ == '__main__':
    path = r'C:\Users\jason\Desktop\0809_C3C4\MTF_Near\1'
    times=2
    findini(path)
    run_looptest(path,times)