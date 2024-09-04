#coding=utf-8
import numpy as np
import cv2
import os
import sys
#from imutils import convenience
#import glob
#import argparse
global tH,tW
import time
sfrimages=[]
images_infile=[]
template=[]
reshape_scale=4
halfboxwidth=40
boxscale=[0.8,0.6,0.4,0.3,0.2]
distance=''
showimagelog=0
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
        #resized = imutils.resize(templatetemp, width=int(templatetemp.shape[1] * scale))
        resized = imutil_resize(templatetemp, width=int(templatetemp.shape[1] * scale))
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
    if showimagelog==1:
        cv2.imshow("template",template)
        cv2.waitKey(0)

def loadallimage(tofsn_path):
    findtype = '.jpg'
    tofsn_path=tofsn_path+'\\'
    files = os.listdir(tofsn_path)
    for file in files:
        # print(file)
        if (file.endswith(findtype)):

            find_imgpath = tofsn_path + '\\' + file
            images_infile.append(find_imgpath)

def read_all_jpg(distance):
    #print(images_infile)
    for file in images_infile:
        #print(file)
        if distance in file:
            sfrimages.append(file)

    #print(new_ini_path)
    #return sfrimages
def newROIlist2(lt_x,lt_y,halfboxwidth):
    ROIlist = ''
    ratio = reshape_scale

    for a in range(1,26):
       if a==1:
           ROIlist=ROIlist+'ROI = {'+"\""+str("%02d"%(a))+str("SFR")+"\": "+'['+str(lt_x * ratio)+', '+str(lt_y * ratio)+', '+str(halfboxwidth * 2 * ratio)+', '+str(halfboxwidth * 2 * ratio)+'], '
       elif a<25:
           ROIlist = ROIlist + "\""+str("%02d"%(a))+str("SFR")+"\": "+'['+str(0)+', '+str(0)+', '+str(0)+', '+str(0)+'], '
       else:
           ROIlist = ROIlist + "\""+str("%02d"%(a))+str("SFR")+"\": "+'['+str(0)+', '+str(0)+', '+str(0)+', '+str(0)+']} '




    # ROIlist="\t\""+str("%02d"%count)+str("SFR")+"\": "+'['+str(lt_x*ratio)+', '+str(lt_y*ratio)+', '+str(halfboxwidth*2*ratio)+', '+str(halfboxwidth*2*ratio)+'], '+'\n'

    return ROIlist
def newROIlist(lt_x,lt_y,halfboxwidth):
    #return str
    #print('halfboxwidth=  '+str(halfboxwidth) )
    ROIlist = ''
    ratio=reshape_scale
    count = 1


    #ROIlist="\t\""+str("%02d"%count)+str("SFR")+"\": "+'['+str(lt_x*ratio)+', '+str(lt_y*ratio)+', '+str(halfboxwidth*2*ratio)+', '+str(halfboxwidth*2*ratio)+'], '+'\n'
    ROIlist = "\t\"" + str("%02d" % count) + str("SFR") + "\": " + '[' + str(lt_x * ratio) + ', ' + str(lt_y * ratio) + ', ' + str(halfboxwidth * 2 * ratio) + ', ' + str(halfboxwidth * 2 * ratio) + '], ' + '\n'
    return ROIlist
def change_number(num):
    count=num-3
    if count==25:
        word = "\t\"" + str("%02d" % count) + str("SFR") + "\": " + '[' + str(0) + ', ' + str(0) + ', ' + str(
            0) + ', ' + str(0) + '] ' + '\n'
    else:
        word="\t\""+str("%02d"%count)+str("SFR")+"\": "+'['+str(0)+', '+str(0)+', '+str(0)+', '+str(0)+'], '+'\n'



    return word
def imutil_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized
def saveboxinfo(lt_x,lt_y,halfboxwidth,file_name,imagename):
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
        file_name = imagename.split(".")[0]
        file_name = file_name.split("/")[-1]
        imname = file_name.split("\\")[-1]
        outputpath = imagename.split(imname)[0]
        #print(outputpath)
        #outputpath=file_name.split("/")[0]

        lines[1] = 'IMG_PATH '+'= '+imagename+'\n'
                #print('=============================================')
                #print(lines[4])
        lines[2] = 'OUTPUT_PATH ' + '= ' + outputpath+'output' + '\n'

        for i in range(3,len(lines)):
            lines[i]=''
        #print(lines[3])
        lines[3] = newROIlist2(lt_x, lt_y, halfboxwidth)

        #lines[4] = newROIlist(lt_x, lt_y, halfboxwidth)

        #for i in range(5,29):
        #    lines[i]=change_number(i)
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
    #print(magnitude_mean.shape)

    bound=10
    #magnitude_col=magnitude[:,int(w_half/2)]
    magnitude_col = magnitude_mean[int(h_half/2), bound:]

    #print(magnitude_col.shape)
    #np.savetxt('magnitude.txt', magnitude_col)
    #np.savetxt('magnitude_col.txt', magnitude_col )
    #print(magnitude[:,int(w_half/2),:])
    #min_valx, max_valx, min_locx, max_locx = cv2.minMaxLoc(grad_x)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(magnitude_col)
    #print(max_locy[0])
    #print(max_locy[1])
    #cv2.circle(half_chartimage,(int(w_half/2),max_locy[1]),5,(0,0,255),4)
    #cv2.imshow("half_chartimage", half_chartimage)
    ref_x = max_loc[1]+bound

    ref_y = int(h_half/2)

    #ref_x=int(w_half/2)

    #ref_y = max_locy[1]

    #print('ref_x= '+str(ref_x)+', ref_y= '+str(ref_y))
    return (ref_x,ref_y)
    #rect_width, rect_height = 50, 50
    #top_leftx = (max(max_locx[0] - rect_width // 2, 0), max(max_locx[1] - rect_height // 2, 0))
    #bottom_rightx = (
    #min(max_locx[0] + rect_width // 2, image.shape[1]), min(max_locx[1] + rect_height // 2, image.shape[0]))
    #cv2.rectangle(image, top_leftx, bottom_rightx, (0, 255, 0), 2)

    #return (int(h_half/2)),

def templatematch():
    global imgcopy,halfboxwidth
    for imagename in sfrimages:
        #print(imagename)
        image=cv2.imread(imagename)
        h, w, _ = image.shape
        imgcopy = image.copy
        image = cv2.resize(image, (int(w / reshape_scale), int(h / reshape_scale)), interpolation=cv2.INTER_AREA)
        #cv2.imshow("image", image)
        #cv2.waitKey(0)
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        found=None

        for scale in  np.linspace(0.2,1.0,20)[::-1]:
            resized=imutil_resize(gray,width = int(gray.shape[1]*scale))
            #resized=imutils.resize(gray,width = int(gray.shape[1]*scale))
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

        #cv2.imwrite('halfchartimage.jpg',half_chartimage)
        (ref_x,ref_y)=chooseSFRROI(half_chartimage)
       # cv2.rectangle(half_chartimage, (ref_x-10, ref_y-10), (ref_x+10, ref_y+10), (0, 0, 255), 2)

        center_x=ref_x+startX
        center_y=ref_y+startY

        file_name = imagename.split(".")[0]
        file_name = file_name.split("/")[-1]
        file_name=file_name.strip('\/')
        print('filename= '+file_name)
        saveboxinfo(center_x-halfboxwidth, center_y-halfboxwidth, halfboxwidth,file_name,imagename)
        if showimagelog == 1:
            cv2.rectangle(image,(startX,startY),(endX,endY),(0,0,255),2)
            cv2.imshow("half_chartimage", half_chartimage)

        #print(file_name)
            cv2.rectangle(image, (int((center_x-halfboxwidth)), int((center_y-halfboxwidth))), (int((center_x+halfboxwidth)), int((center_y+halfboxwidth))), (255, 0, 0), 2)
            #cv2.circle(image, (lt_x, lt_y), 5, (0, 0, 255), 4)
            cv2.imshow("image", image)
            cv2.imwrite('output/'+file_name+'_result.jpg', image)
            cv2.waitKey(0)


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


def cleanimage():
    global sfrimages, template,halfboxwidth
    sfrimages = []
    template = []
    halfboxwidth = 40
if __name__ == '__main__':

    tofsn_path = sys.argv[1]
    #tofsn_path = r'C:\Users\jason\Desktop\0809_C3C4\C5\CO1-17ABOK-00100_5\CO1-17ABOK-00100'
    #tofsn_path=r'C:\Users\jason\Desktop\0809_C3C4\C5\CO1-17ABOK-00078_2\CO1-17ABOK-00078'
    startime=time.time()
    #loadallimage(r'C:\Users\jason\PycharmProjects\pythonProject\SFRdetect_new')
    loadallimage(tofsn_path)
    dis=['1m','2m','3m','4m','5m']
    #dis=['1m']
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
