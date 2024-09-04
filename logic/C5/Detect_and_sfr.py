import numpy as np
import os
import time
import encodings
import shutil
#current_path=os.getcwd()+'//'
import sys
from os import walk
import subprocess
new_ini_path=[]


if __name__ == '__main__':

    #rootpath = r"C:\Users\jason\PycharmProjects\pythonProject\SFRdetect_new"
    #rootpath=r"C:\Users\jason\Desktop\logic_Calibration\chriss\SN123456_test1"
    rootpath = sys.argv[1]
    print("ROI detect go all folder under root: "+rootpath)
    #print(tofsn_path)
    #current_path = os.getcwd()
    print("===執行DetectROI.py===")


    for folder, dirs, files in walk(rootpath):

        print('current folder ===>' + folder)


        findtype = '.ini'

        hasini=0
        #if(files):
        for name in files:
            if name.endswith(findtype):
                hasini=1
        #print(files)
        if (hasini==1 and 'output' not in folder and 'tmp' not in folder ):

            print("=========執行中=========")
            os.system("DetectROI.exe "+folder)



    print("Complete DetectROI")

    print("===執行RunLogicsft.exe===")
    os.system("RunLogicsfr.exe " + rootpath)
    '''
    
    for root, dirs, files in walk(rootpath):


        print("路徑：", root)
        #autoroi_cmd = ["gen_sfr_result.exe", root]

        findtype = '.ini'

        hasini = 0
        # if(files):
        for name in files:
            if name.endswith(findtype):
                hasini = 1
        # print(files)
        if (hasini == 1 and 'output' not in root):



            print("=========執行中=========")
            os.system("gen_sfr_result.exe " + root)

    '''
