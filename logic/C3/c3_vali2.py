global tH,tW
import time
#sfrimages=[]
import statistics
import numpy as np
import cv2
from sklearn.cluster import KMeans
csv_name=[]
shift_outimage_list=[]
outbound_list=[]
csv_folder=[]

showimagelog=1
global imgcopy
import os
import pandas as pd
column_name=['SN','imagename','01SFR','02SFR','03SFR','04SFR','05SFR','06SFR','10SFR','11SFR','15SFR','16SFR','20SFR','21SFR','22SFR','23SFR','24SFR','25SFR']
origindata=pd.DataFrame(columns=column_name)
col=['SN','lx','ly','cx','cy','outofimage']
ROI1=pd.DataFrame(columns=col)

def readcsv_to_dataframe(csv):
    global origindata

    csv_dataframe = pd.read_csv(csv, delimiter=';')
    csv_dataframe.columns = csv_dataframe.columns.str.strip(',')
    #print(csv_dataframe.columns)
    csv_dataframe = csv_dataframe.drop(csv_dataframe[csv_dataframe["Match States"] == "NG"].index)
    # print(csv_dataframe.index)
    csv_dataframe = csv_dataframe.reset_index()
    # print(csv_dataframe.index)
    # csv_dataframe.to_csv('test.csv', encoding='utf-8', index=False)

    datasize = csv_dataframe.shape[0]
    nametemp_name = []
    # print(nametemp)

    #for img in nametemp:

        #nametemp_path = img.split("\\")[-1]
        #nametemp_name.append(nametemp_path)
    # print(nametemp_name)
    # print(datasize)
    # print(sfrimages_name)
    # print(csv_dataframe.columns)
    SNname_set = []
    for i in range(0, datasize):
        # print(i)
        file_name_in_csv = csv_dataframe.loc[i, 'File Name']
        SN_name = file_name_in_csv.split("_")[0]
        # print('SN_name= '+SN_name)
        # find image name from img to csv
        # print(sfrimages)
        if SN_name not in SNname_set :
            # print('add '+file_name_in_csv)
            #sfrimages.append(file_name_in_csv)
            SNname_set.append(SN_name)
            # print("file_name_in_csv")
            # print(file_name_in_csv)
            ROI_Coordinate = csv_dataframe.loc[i, 'ROI Coordinate']
            ROI_Coordinate = ROI_Coordinate.split("],")
            #print('---------')
            origindata = origindata._append({"SN": SN_name,"imagename":file_name_in_csv},ignore_index=True)
            #print(origindata.shape[0])
            current_outputrow=origindata.shape[0]-1
            current_outputcol=2
            for j in ROI_Coordinate:
                for column_list in column_name:
                    if column_list in j:
                        ROI_list_temp=j.split("[")[1].strip(']}')
                        #coord_ROI.append(ROI_list_temp)
                        origindata.iloc[current_outputrow, current_outputcol]=ROI_list_temp
                        current_outputcol=current_outputcol+1
    newcol=['SNname','imagename','ROI_01','ROI_02','ROI_03','ROI_04','ROI_05','ROI_06','ROI_10','ROI_11','ROI_15','ROI_16','ROI_20','ROI_21','ROI_22','ROI_23','ROI_24','ROI_25']
    origindata.columns=newcol
    print(origindata)
                #if '13SFR' in j:
                #    pos = j.split(",")
                #    lx = pos[0].split("[")[-1]
                #    ly = pos[1]
                #    w = pos[2]
                #    h = pos[3]
                #    cx = int(lx) + int(int(w) / 2)
                #    cy = int(ly) + int(int(h) / 2)
                #    # sn_name=file_name_in_csv.split("_")[0]
                #    outputdata = outputdata._append({"SN": SN_name, "cx_roddick": cx, "cy_roddick": cy},ignore_index=True)
def parse_all_csv(Roi_data_path):
    global csv_name,csv_folder
    csvtype = '.csv'
    path = Roi_data_path + '\\'
    for folder, subfolder, files in os.walk(path):
        if '_internal' in folder:
            continue

        for file in files:

        #set_id=set(file_split[0])
        #print(set_id)
            if (file.endswith(csvtype) and 'Output' in file):
                output_folder = folder +'\\'+ 'output'
                try:
                    #print("make output folder")
                    os.makedirs(output_folder)
                except FileExistsError:
                    pass
                csvpath=folder+'\\'+file
                csv_folder.append(folder)
                csv_name.append(csvpath)
    print(csv_name)
def bbox_judge(roi):
    group=3


    gray = roi
    #cv2.imshow("gray",gray)
    #cv2.waitKey(0)
    gray = gray / 255.0

    #print(gray.shape)
    #print(gray)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=1)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=1)

    magnitude, angle = cv2.cartToPolar(sobelx, sobely, angleInDegrees=True)
    w, h = magnitude.shape
    # print(angle)
    mag = magnitude.reshape(w * h, 1)
    ang = angle.reshape(w * h, 1)

    kmeans = KMeans(n_clusters=group, random_state=0).fit(mag)
    a = kmeans.labels_.shape

    kmeansangle = np.zeros(3)
    mag0 = []
    mag1 = []
    mag2 = []
    angle0 = []
    angle1 = []
    angle2 = []

    for i in range(0, a[0]):
        if kmeans.labels_[i] == 0:
            mag0.append(mag[i])
            angle0.append(ang[i])
        if kmeans.labels_[i] == 1:
            mag1.append(mag[i])
            angle1.append(ang[i])
        if kmeans.labels_[i] == 2:
            mag2.append(mag[i])
            angle2.append(ang[i])
    count0=len(mag0)
    count1 = len(mag1)
    count2 = len(mag2)
    #print(len(mag0))
    #print(len(mag1))
    #print(len(mag2))
    m0 = np.median(mag0)
    m1 = np.median(mag1)
    m2 = np.median(mag2)
    #print('m0= ' + str(m0 ) + ',m1= ' + str(m1 ) + ',m2= ' + str(m2 ))
    a0 = np.median(angle0)
    a1 = np.median(angle1)
    a2 = np.median(angle2)
    #print('a0= ' + str(a0) + ',a1= ' + str(a1) + ',a2= ' + str(a2))
    comparelist=[]
    countlist=[count0,count1,count2]
    anglelist=[a0,a1,a2]
    maglist = [m0, m1, m2]
    sumcount = sum(countlist)
    for i in range(0,group):
        if countlist[i]/sumcount<0.7 and countlist[i]>10:
            comparelist.append(i)
    if comparelist==[]:
        return True
    for item in comparelist:
        print('comparelist angle')
        print(anglelist[item])
        if anglelist[item]<170 or maglist[item]<0.2:
            print('find')
            return True
    return False
def box_outpattern(folder):
    global ROI1
    origindata_rowsize, origindata_colsize = origindata.shape
    outbound=0
    for i in range(0, origindata_rowsize):
        if i in shift_outimage_list:
            continue
        roiimage=origindata.iloc[i,1]
        roiimage_name=roiimage[:-4]+'_ROI.jpg'
        impath=folder+'//'+roiimage_name
        print('current handle image= ' + impath)
        image = cv2.imread(impath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for j in range(2,origindata_colsize):
            print('j= '+str(j))
            coord=origindata.iloc[i,j]
            coordlist=coord.split(",")
            lx=int(coordlist[0])
            ly=int(coordlist[1])
            w=int(coordlist[2])
            h=int(coordlist[3])
            #if(j==2):
            #    ROI1=ROI1._append({'SN':origindata.iloc[i,0],'lx':lx,'ly':ly,'cx':str(lx+w/2),'cy':str(ly+h/2),'outofimage':'False'},ignore_index=True)
            #    if i in shift_outimage_list:
            #        ROI1.loc[i,'outofimage']=True
            scale_x=6.72
            scale_y=6.66
            roi_ly=int(ly/scale_y)+41
            roi_lx=int(lx/scale_x)+56
            print('roi_ly= '+str(roi_ly)+',roi_lx= '+str(roi_lx))
            roi_w=30
            roi_h=30
            roi=image[roi_ly:roi_ly+roi_h,roi_lx:roi_lx+roi_w]
            bound = 3
            #roi = roi[bound:roi_h - bound, bound:roi_w - bound]
            #cv2.imshow('ROI',roi)
            #cv2.waitKey(0)
            #boundresult=bbox_judge(roi)
            #outbound=outbound|boundresult
            if outbound==1:
                print(str(i)+'!!!!!out of bound')
                outbound_list.append(i)
                break
    #ROI1.to_csv(r'C:\Users\jason\Desktop\0809_C3C4\0823_C3\ROI1_5m.csv', encoding='utf-8', index=False)
    print('outbound_list=')
    print(outbound_list)
def box_outofimg():
    global shift_outimage_list
    boundmin_x=0
    boundmin_y=0
    boundmax_x = 5472
    boundmax_y = 3078
    origindata_rowsize,origindata_colsize=origindata.shape
    print(origindata_rowsize)
    lx_list=[]
    ly_list=[]
    for i in range(0, origindata_rowsize):
        for j in range(2,origindata_colsize):
            coord=origindata.iloc[i,j]
            coordlist=coord.split(",")
            lx=int(coordlist[0])
            ly=int(coordlist[1])
            w=int(coordlist[2])
            h=int(coordlist[3])
            if j==2:
                lx_list.append(lx)
                ly_list.append(ly)

            if lx<boundmin_x or ly<boundmin_y or lx+w>boundmax_x or ly+h>boundmax_y:
                #print(origindata.iloc[i,1])
                shift_outimage_list.append(i)
                break
    #threesigma(lx_list,ly_list)
    print(shift_outimage_list)
def threesigma(lx_list,ly_list):
    boundmin_x = 0
    boundmin_y = 0
    boundmax_x = 5472
    boundmax_y = 3078
    meanx = np.mean([x for x in lx_list if x>boundmin_x and x<boundmax_x])
    meany = np.mean([y for y in ly_list if y>boundmin_y and y<boundmax_y])
    stdx=np.std([x for x in lx_list if x>boundmin_x and x<boundmax_x])
    stdy=np.std([y for y in ly_list if y>boundmin_y and y<boundmax_y])

    low3sigma_x=meanx-3*stdx
    high3sigma_x=meanx+3*stdx
    low3sigma_y = meany - 3 * stdy
    high3sigma_y = meany + 3 * stdy
    xmuch=[index for index, value in enumerate(lx_list) if value > high3sigma_x]
    xless=[index for index, value in enumerate(lx_list) if value < low3sigma_x]
    ymuch = [index for index, value in enumerate(ly_list) if value > high3sigma_y]
    yless = [index for index, value in enumerate(ly_list) if value < low3sigma_y]
    print(low3sigma_y)
    print(high3sigma_y)
    print('*****')
    print(yless)
    print(ymuch)
    threesigma_list=set(xmuch) |set(xless) |set(ymuch) |set(yless)
    threesigma_list=list(threesigma_list)
    threesigma_list.sort()
    print('three sigma')
    print(threesigma_list)
    print('three sigma end')

if __name__ == '__main__':
    Roi_data_path = r'C:\Users\jason\Desktop\0809_C3C4\0823_C3'
    startime = time.time()
    parse_all_csv(Roi_data_path)
    # for csv in csv_name:
    #    readcsv_to_dataframe(csv)
    readcsv_to_dataframe(csv_name[0])
    box_outofimg()
    box_outpattern(csv_folder[0])
    #box_outpattern(csv_name[0])
    #origindata.to_csv(r'C:\Users\jason\Desktop\0809_C3C4\0823_C3\test5m.csv', encoding='utf-8', index=False)

