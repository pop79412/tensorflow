#coding=utf-8
import numpy as np
import cv2
import os
import sys
import imutils
import csv
import math
#import glob
#import argparse
global tH,tW
import time
sfrimages=[]
images_infile=[]
template=[]
csv_name=[]
nametemp = []
reshape_scale=4
halfboxwidth=40
import pandas as pd
outputdata=pd.DataFrame({'SN':[],'cx_ref':[],'cy_ref':[],'cx_roddick':[],'cy_roddick':[],'Euclidean Distance':[]})

boxscale=[1,0.6,0.4,0.3,0.2]
distance=''
showimagelog=1
global imgcopy
def dinamic_boxsize(distance):

    global halfboxwidth
    if distance=='1m':
        halfboxwidth=int(halfboxwidth*boxscale[0])
    if distance=='2m':
        halfboxwidth=int(halfboxwidth*boxscale[1])
    if distance=='3m':
        halfboxwidth=int(halfboxwidth*boxscale[2])
    if distance=='4m':
        halfboxwidth=int(halfboxwidth*boxscale[3])
    if distance=='5m':
        halfboxwidth=int(halfboxwidth*boxscale[4])
def loadtemplate_multiscale():
    global template, tH, tW
    tempath = 'template/template.jpg'
    templatetemp = cv2.imread(tempath)
    h, w, _ = templatetemp.shape
    templatetemp = cv2.resize(templatetemp, (int(w / reshape_scale), int(h / reshape_scale)), interpolation=cv2.INTER_AREA)
    (tH, tW) = templatetemp.shape[:2]
    templatetemp = cv2.cvtColor(templatetemp, cv2.COLOR_BGR2GRAY)
    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        resized = imutils.resize(templatetemp, width=int(templatetemp.shape[1] * scale))

        #templatetemp = cv2.Canny(templatetemp, 50, 200)
        templatetemp = cv2.Canny(resized, 50, 200)
        template.append(templatetemp)
def loadtemplate(distance):
    global template,tH,tW
    tempath='template/'+distance+'_template.jpg'
    template=cv2.imread(tempath)

    h, w, _ = template.shape

    template = cv2.resize(template, (int(w / reshape_scale), int(h / reshape_scale)), interpolation=cv2.INTER_AREA)
    template=cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
    template=cv2.Canny(template,50,200)
    (tH,tW)= template.shape[:2]
    #if showimagelog==1:
    #    cv2.imshow("template",template)
    #    cv2.waitKey(0)

def loadallimage(tofsn_path):
    findtype = '.jpg'
    tofsn_path=tofsn_path+'\\'
    files = os.listdir(tofsn_path)
    for file in files:
        # print(file)
        if (file.endswith(findtype)):

            find_imgpath = tofsn_path + '\\' + file
            images_infile.append(find_imgpath)
def load_unique_image(tofsn_path):
    findtype = '.jpg'
    tofsn_path=tofsn_path+'\\'
    files = os.listdir(tofsn_path)
    csvtype='.csv'
    output_folder=tofsn_path+'output'
    print('output folder= '+output_folder)
    #if os.path.exists(output_folder):
    try:
        print("make output folder")
        os.makedirs(output_folder)
    except FileExistsError:
    #catch:
        print("exist outputfolder")
    #    pass

    for file in files:
        # print(file)

        file_split=file.split("_")
        find_imgpath = tofsn_path + '\\' + file
        #set_id=set(file_split[0])
        #print(set_id)
        if (file.endswith(csvtype) ):
            csv_name.append(find_imgpath)
        if (file.endswith(findtype) and file_split[0] not in nametemp):
            #if showimagelog == 1:
            #    print(file_split[0])
            #find_imgpath = tofsn_path + '\\' + file
            #print(find_imgpath)
            sfrimages.append(find_imgpath)
            nametemp.append(file_split[0])

def read_all_jpg(distance):
    #print(images_infile)
    for file in images_infile:
        #print(file)
        if distance in file:
            sfrimages.append(file)

    #print(new_ini_path)
    #return sfrimages

def newROIlist(lt_x,lt_y,halfboxwidth):
    #return str
    #print('halfboxwidth=  '+str(halfboxwidth) )
    ROIlist = ''
    ratio=reshape_scale
    count = 1

    #ROIlist=ROIlist+"\t"
    ROIlist="\t\""+str("%02d"%count)+str("SFR")+"\": "+'['+str(lt_x*ratio)+', '+str(lt_y*ratio)+', '+str(halfboxwidth*2*ratio)+', '+str(halfboxwidth*2*ratio)+'], '+'\n'
    #count = count + 1




    return ROIlist
def saveboxinfo(lt_x,lt_y,halfboxwidth,file_name):
    global imgcopy
    ratio=reshape_scale
    count = 1
    path = str(file_name)+'.ini'
    #print('ratio= '+str(ratio) )

    #if os.path.exists(path):

        #os.chmod(path, 0o666)
        #print("File permissions modified successfully!")
    #else:
        #print("File not found:", path)
    with open(path, 'r') as f1:
        lines = f1.readlines()
        keyword = '"01SFR":'
        keyword2 = 'IMG_PATH'
        #print("===walk in ini file===")
        #for line in lines:
            #string = line.split(' ')
            #print(string)

        #    if keyword in string:
                #print(string)
                # saveboxinfo(lt_x,lt_y,halfboxwidth)
                # line=line.replace(string,newROIlist(lt_x,lt_y,halfboxwidth))
        lines[4] = newROIlist(lt_x, lt_y, halfboxwidth)

                #print('=============================================')
                #print(lines[4])
        lines[2] = 'OUTPUT_PATH ' + '=' + ' output' + '\n'
        newpath = file_name+'.ini'
        with open(newpath, 'w') as f2:
            f2.writelines(lines)
        print("===finish write  "+file_name+".ini ===")
        #f.write('test  [' + str(lt_x) + ', ' + str(lt_y)  + '], ')
        #cv2.rectangle(imgcopy, (lt_x*ratio, lt_y*ratio),(lt_x*ratio+halfboxwidth*2*ratio, lt_y*ratio+halfboxwidth*2*ratio), (0, 0, 255), 2*ratio)
        #cv2.imwrite('output_resize.jpg',img)
        #cv2.imwrite('output.jpg', imgcopy)
def chooseSFRROI(half_chartimage):
    h_half, w_half, _ = half_chartimage.shape
    #print('h_half= '+str(h_half)+' w_half= '+str(w_half))
    #print(half_chartimage.shape)
    #ret, half_chartimage = cv2.threshold(half_chartimage, 127, 255, cv2.THRESH_BINARY)
    #half_chartimage=half_chartimage.mean()
    grad_x = cv2.Sobel(half_chartimage, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(half_chartimage, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = abs(grad_x) + abs(grad_y)
    magnitude_mean = magnitude.mean(2)
    #print(grad_x.shape)
    #print(grad_y.shape)
    #print(magnitude_mean.shape)


    bound=10
    magnitude_col=magnitude[bound:,int(w_half/2)]
    #np.savetxt('magnitude.txt', magnitude_mean)
    #np.savetxt('magnitude_col.txt', magnitude_col )
    #print(magnitude[:,int(w_half/2),:])
    #min_valx, max_valx, min_locx, max_locx = cv2.minMaxLoc(grad_x)

    min_valy, max_valy, min_locy, max_locy = cv2.minMaxLoc(magnitude_col)
    #print(max_locy[0])
    #print(max_locy[1])
    #cv2.circle(half_chartimage,(int(w_half/2),max_locy[1]),5,(0,0,255),4)
    #cv2.imshow("half_chartimage", half_chartimage)
    ref_x=int(w_half/2)
    ref_y=max_locy[1]+bound
    #print('ref_x= '+str(ref_x)+', ref_y= '+str(ref_y))
    return (ref_x,ref_y)
    #rect_width, rect_height = 50, 50
    #top_leftx = (max(max_locx[0] - rect_width // 2, 0), max(max_locx[1] - rect_height // 2, 0))
    #bottom_rightx = (
    #min(max_locx[0] + rect_width // 2, image.shape[1]), min(max_locx[1] + rect_height // 2, image.shape[0]))
    #cv2.rectangle(image, top_leftx, bottom_rightx, (0, 255, 0), 2)

    #return (int(h_half/2)),

def templatematch(distance,tofsn_path):
    global imgcopy,outputdata
    count=0
    for imagename in sfrimages:
        #print(imagename)
        image_path=tofsn_path+'/'+imagename
        #print(image_path)
        image=cv2.imread(image_path)
        h, w, _ = image.shape
        imgcopy = image.copy
        image = cv2.resize(image, (int(w / reshape_scale), int(h / reshape_scale)), interpolation=cv2.INTER_AREA)

        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        found=None

        for scale in  np.linspace(0.2,1.0,20)[::-1]:
            resized=imutils.resize(gray,width = int(gray.shape[1]*scale))
            r=gray.shape[1] /float(resized.shape[1])

            if resized.shape[0] <tH or resized.shape[1] < tW:
                break



            edged=cv2.Canny(resized,50,200)
            #cv2.imshow("edged", edged)
            #cv2.waitKey(0)


            result=cv2.matchTemplate(edged,template,cv2.TM_CCOEFF)
            (_,maxVal, _, maxLoc)=cv2.minMaxLoc(result)

            #for templates in template:
            #    result = cv2.matchTemplate(edged, templates, cv2.TM_CCOEFF)
            #    loc = np.where(res >= threshold)
            #result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
            #(_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

            #if args.get("visualize",False):
            #clone=np.dstack([edged,edged,edged])
            #cv2.rectangle(clone,(maxLoc[0],maxLoc[1]),(maxLoc[0]+tW,maxLoc[1]), (0, 0, 255),4)
            #cv2.imshow("Visualize",clone)
            #cv2.waitKey(0)
            (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
            (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
            box_w=endX-startX
            box_h=endY-startY
            if found is None or maxVal > found[0] and box_w<tW/reshape_scale and box_h<tH/reshape_scale:
                found= (maxVal,maxLoc,r)

            #if found is None or maxVal > found[0]:
            #    found = (maxVal, maxLoc, r)

        (_,maxLoc,r)=found
        (startX,startY)=(int(maxLoc[0]*r),int(maxLoc[1]*r))
        (endX,endY)=(int((maxLoc[0]+tW)*r),int((maxLoc[1]+tH)*r))


        centerX = int((startX+endX)/2)
        centerY = int((startY + endY) / 2)
        #saveboxinfo(centerX, centerY, halfboxwidth)
        chartimage=image[startY:endY,startX:endX,:]
        #cv2.imshow("chartimage", chartimage)
        half_chartimage=image[startY:endY,startX:int( (endX+startX)/2),:]
        #cv2.imshow("half_chartimage", half_chartimage)
        #(ref_x,ref_y)=chooseSFRROI(half_chartimage)
        #center_x=ref_x+startX
        #center_y=ref_y+startY
        #scale_halfboxwidth_x=int(halfboxwidth_x/reshape_scale)
        #scale_halfboxwidth_y =int( halfboxwidth_y / reshape_scale)
        center_x =  centerX*reshape_scale
        center_y =  centerY*reshape_scale
        #outputdata=outputdata._append({"SN":nametemp[count],"cx_ref":center_x,"cy_ref":center_y},ignore_index=True)
        #print(outputdata)
        sn_name=imagename.split("_")[0]
        outputdata.loc[outputdata["SN"] == sn_name, 'cx_ref'] = center_x
        outputdata.loc[outputdata["SN"] == sn_name, 'cy_ref'] = center_y
        #ref_cx=outputdata.loc[outputdata["SN"]==sn_name,'cx_ref']
        #ref_cy=outputdata.loc[outputdata["SN"]==sn_name,'cy_ref']
        # dif=int(math.sqrt( (ref_cx-cx)**2+(ref_cy-cy)**2 ))
        # outputdata.loc[outputdata["SN"] == sn_name, 'Euclidean Distance'] = dif
        # outputdata.loc[outputdata["SN"]==sn_name,'cx_roddick']=cx
        # outputdata.loc[outputdata["SN"]==sn_name, 'cy_roddick'] = cy




        #file_name = imagename.split(".")[0]

        #file_name = file_name.split("/")[-1]
        #file_name=file_name.strip('\/')
        #print('filename= '+file_name)
        print('imagename= ' + imagename)



        #cv2.imwrite('output/' + file_name + '_result.jpg', image)
        count=count+1

        #saveboxinfo(center_x-halfboxwidth, center_y-halfboxwidth, halfboxwidth,file_name)
        if showimagelog == 1:
            #cv2.imshow("image", image)
            #cv2.waitKey(0)
            #cv2.rectangle(image,(startX,startY),(endX,endY),(0,0,255),2)
            #cv2.imshow("chartimage", chartimage)

        #print(file_name)
            cv2.rectangle(image, (int((startX)), int((startY))), (int((endX)), int((endY))), (255, 0, 0), 1)
            cv2.circle(image, (centerX, centerY), 1, (0, 0, 255), 4)
            #cv2.imshow("image", image)
            #snname=file_name.split('\\')[-1]
            output_folder=tofsn_path+'/output/'

                #print("exist outputfolder")
            outputpath=output_folder+sn_name+'_result.jpg'
            #print(outputpath)
            cv2.imwrite(outputpath, image)
            #print(outputpath)
            #cv2.imwrite(file_name+'_result.jpg', image)
            #cv2.waitKey(0)


        #cv2.imwrite('output/' + 'chartimage.jpg', chartimage)
        #criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
        #compactness, clusters, centers = cv2.kmeans(chartimage, K=3, bestLabels=None, criteria=criteria,attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)
        #centers = np.uint8(centers)
        #print(centers)
        #cv2.waitKey(0)




def templatematch2():
    for image in sfrimages:
        image=cv2.imread(image)
        #imgcopy=image.copy
        h, w, _ = image.shape
        #image = cv2.resize(image, (int(w / 4), int(h / 4)), interpolation=cv2.INTER_AREA)
        #cv2.imshow("image", image)
        #cv2.waitKey(0)
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        found=None

        for scale in  np.linspace(0.2,1.0,20)[::-1]:
            resized=imutils.resize(gray,width = int(gray.shape[1]*scale))
            r=gray.shape[1] /float(resized.shape[1])

            if resized.shape[0] <tH or resized.shape[1] < tW:
                break



            edged=cv2.Canny(resized,50,200)
            #cv2.imshow("edged", edged)
            #cv2.waitKey(0)


            #result=cv2.matchTemplate(edged,template,cv2.TM_CCOEFF)
            #(_,maxVal, _, maxLoc)=cv2.minMaxLoc(result)

            for templates in template:
                result = cv2.matchTemplate(edged, templates, cv2.TM_CCOEFF)
                #loc = np.where(res >= threshold)
                (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
            #result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
            #(_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

            #if args.get("visualize",False):
            clone=np.dstack([edged,edged,edged])
            #cv2.rectangle(clone,(maxLoc[0],maxLoc[1]),(maxLoc[0]+tW,maxLoc[1]), (0, 0, 255),4)
            #cv2.imshow("Visualize",clone)
            #cv2.waitKey(0)

            if found is None or maxVal > found[0]:
                found= (maxVal,maxLoc,r)

        (_,maxLoc,r)=found
        (startX,startY)=(int(maxLoc[0]*r),int(maxLoc[1]*r))
        (endX,endY)=(int((maxLoc[0]+tW)*r),int((maxLoc[1]+tH)*r))

        cv2.rectangle(image,(startX,startY),(endX,endY),(0,0,255),2)
        cv2.imshow("image",image)
        cv2.waitKey(0)
def load_image_and_csv(tofsn_path):
    findtype = '.jpg'
    tofsn_path=tofsn_path+'\\'
    files = os.listdir(tofsn_path)
    csvtype='.csv'
    output_folder=tofsn_path+'output'
    print('output folder= '+output_folder)
    #if os.path.exists(output_folder):
    try:
        print("make output folder")
        os.makedirs(output_folder)
    except FileExistsError:
    #catch:
        print("exist outputfolder")
    #    pass

    for file in files:
        # print(file)

        file_split=file.split("_")
        find_imgpath = tofsn_path + '\\' + file
        #set_id=set(file_split[0])
        #print(set_id)
        if (file.endswith(csvtype) ):
            csv_name.append(find_imgpath)
        #if (file.endswith(findtype) and file_split[0] not in nametemp):
        if (file.endswith(findtype) ):
            #sfrimages.append(find_imgpath)
            nametemp.append(find_imgpath)
def readcsv():
    global outputdata,nametemp
    csv_name_str = ''.join(csv_name)
    csv_dataframe = pd.read_csv(csv_name_str, delimiter=';')
    csv_dataframe.columns = csv_dataframe.columns.str.strip(',')
    csv_dataframe = csv_dataframe.drop(csv_dataframe[csv_dataframe["Match States"]=="NG"].index)
    #print(csv_dataframe.index)
    csv_dataframe=csv_dataframe.reset_index()
    #print(csv_dataframe.index)
    #csv_dataframe.to_csv('test.csv', encoding='utf-8', index=False)

    datasize=csv_dataframe.shape[0]
    nametemp_name=[]
    #print(nametemp)

    for img in nametemp:
        #sfrimages_path = img.split(".")[0]
        nametemp_path = img.split("\\")[-1]
        nametemp_name.append(nametemp_path)
    #print(nametemp_name)
    #print(datasize)
    #print(sfrimages_name)
    #print(csv_dataframe.columns)
    SNname_set=[]
    for i in range(0,datasize):
        #print(i)
        file_name_in_csv=csv_dataframe.loc[i,'File Name']
        SN_name = file_name_in_csv.split("_")[0]
        #print('SN_name= '+SN_name)
        #find image name from img to csv
        #print(sfrimages)
        if SN_name not in SNname_set and file_name_in_csv in nametemp_name:
            #print('add '+file_name_in_csv)
            sfrimages.append(file_name_in_csv)
            SNname_set.append(SN_name)
            #print("file_name_in_csv")
            #print(file_name_in_csv)
            ROI_Coordinate=csv_dataframe.loc[i, 'ROI Coordinate']
            ROI_Coordinate=ROI_Coordinate.split("],")
            for j in ROI_Coordinate:
                if '13SFR' in j:
                    pos=j.split(",")
                    lx=pos[0].split("[")[-1]
                    ly=pos[1]
                    w=pos[2]
                    h=pos[3]
                    cx=int(lx)+int( int(w)/2)
                    cy=int(ly)+int(int(h)/2)
                    #sn_name=file_name_in_csv.split("_")[0]
                    outputdata = outputdata._append({"SN": SN_name, "cx_roddick": cx, "cy_roddick": cy},
                                                   ignore_index=True)
                    #print(outputdata)
                    #ref_cx=outputdata.loc[outputdata["SN"]==sn_name,'cx_ref']
                    #ref_cy=outputdata.loc[outputdata["SN"]==sn_name,'cy_ref']
                    #dif=int(math.sqrt( (ref_cx-cx)**2+(ref_cy-cy)**2 ))
                    #outputdata.loc[outputdata["SN"] == sn_name, 'Euclidean Distance'] = dif
                    #outputdata.loc[outputdata["SN"]==sn_name,'cx_roddick']=cx
                    #outputdata.loc[outputdata["SN"]==sn_name, 'cy_roddick'] = cy
    #csv_name_str=''.join(csv_name)
    #print(str1)
    #with open(csv_name_str) as csv_read_file:
    #    csv_read_data = csv.reader(csv_read_file)
    #    csv_read_list = list(csv_read_data)
    #    for row in csv_read_list:
    #        print(row)
def countdiff():
    global outputdata
    datasize = outputdata.shape[0]
    #print('datasize= '+str(datasize))
    for i in range(0, datasize):


        ref_cx=outputdata.loc[i,'cx_ref']
        ref_cy=outputdata.loc[i,'cy_ref']
        cx = outputdata.loc[i, 'cx_roddick']
        cy = outputdata.loc[i, 'cy_roddick']
        dif=int(math.sqrt( (ref_cx-cx)**2+(ref_cy-cy)**2 ))
        outputdata.loc[i, 'Euclidean Distance'] = dif

def cleanimage():
    global sfrimages, template,outputdata,csv_name
    sfrimages = []
    template = []
    csv_name=[]

    outputdata = pd.DataFrame({'SN': [], 'cx_ref': [], 'cy_ref': [], 'cx_roddick': [], 'cy_roddick': [],'Euclidean Distance':[]})
if __name__ == '__main__':
    #tofsn_path=r'C:\Users\jason\Desktop\0809_C3C4\origin_image\1m'
    distance='5m'
    tofsn_path = r'C:\Users\jason\Desktop\0809_C3C4\origin_image\5m'
    #tofsn_path = sys.argv[1]
    startime=time.time()
    #loadallimage(tofsn_path)
    #dis=['1m','2m','3m','4m','5m']
    #dis=['5m']

    #distance='1m'
    load_image_and_csv(tofsn_path)
    readcsv()


    #load_unique_image(tofsn_path)
    loadtemplate(distance)
    templatematch(distance,tofsn_path)
    countdiff()
    print(outputdata)
    #outputdata.to_csv('test.csv', encoding='utf-8', index=False)
    #outputdata.to_csv(distance+'.csv', encoding='utf-8', index=False)
    cleanimage()

    '''
    for a in dis:
        #print(a)
        cleanimage()
        distance=a
        print('distance= '+ distance)
        dinamic_boxsize(distance)
        loadtemplate(distance)
    #loadtemplate_multiscale()
        read_all_jpg(distance)
    #print(sfrimages)

        templatematch()
        #cleanimage()
    endtime = time.time()
    
    print('Done')
    print('Total run time= '+str(endtime-startime)+' s')
    '''