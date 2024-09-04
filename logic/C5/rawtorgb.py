import cv2
import numpy as np


import sys




import os

imagelist=[]


def raw16_to_raw12(data):
    size_t = data.shape
    size=int(size_t[0])
    #print('size= '+str(size))
    depthimage=(np.zeros(  int(size/2)))
    confidence=(np.zeros(int(size/2)))
    imageposition=0
    tempcount=0
    for i in range(0,size):
        #print(i)
        if i%2==0:
            tempcount=data[i]

        else:

            #tempcount = tempcount + (((data[i]) ) << 8)
            #depthimage[int((i - 1) / 2)] = tempcount



            tempcount=tempcount+( ((data[i])&(31))<<8 )
            depthimage[int((i-1)/2)]=tempcount
            if ((data[i])>>5)==0:
                confidence[int((i - 1) / 2)] =255
            else:
                confidence[int((i - 1) / 2)]=0
            #confidence[int((i - 1) / 2)] = (data[i]) >> 5
            #print('data[i]= ', str(data[i]))
            #print('tempcount= ', str(tempcount))


    return (depthimage,confidence)

def cleansettings():
    global imagelist
    imagelist=[]
def create_outputfolder(folder):
    irfolder = folder + '\\' + 'ir_image'
    depthfolder = folder + '\\' + 'depth_image'
    if os.path.exists(irfolder):
        pass
    else:
        os.makedirs(irfolder)
    if os.path.exists(depthfolder):
        pass
    else:
        os.makedirs(depthfolder)
def process_raw(folder):
    h = 480
    w = 640

    for imageraw in imagelist:
        if 'ir' in imageraw.lower():
            print('processing: '+imageraw)
            irim = np.fromfile(imageraw, dtype='uint8')
            # ir, confidence_ir = raw16_to_raw8(irim)
            irim = irim.reshape(h, w, 1)
            filename = imageraw.split("\\")[-1].split('.')[0]
            savefilename = folder + '\\' + 'ir_image' + '\\' + filename+'.jpg'
            print('outputname= ' + savefilename)
            cv2.imwrite(savefilename, irim)

        if 'depth' in imageraw.lower():
            print('processing: '+imageraw)
            depth = np.fromfile(imageraw, dtype='uint8')
            dep, confidence = raw16_to_raw12(depth)
            dep = dep.reshape(h, w, 1)
            filename = imageraw.split("\\")[-1].split('.')[0]
            savefilename=folder+'\\'+'depth_image'+'\\'+filename+'.jpg'
            print('outputname= '+savefilename)
            cv2.imwrite(savefilename, dep)


if __name__ == '__main__':
    path = r'C:\Users\jason\Desktop\0809_C3C4\C1\279_tof_camera\CO1-16SBOK-00001'
    #path = sys.argv[1]
    #choose_type = sys.argv[2]
    #path = r'C:\Users\jason\Desktop\0809_C3C4\C5\CO1-17ABOK-00078_1'
    path = path + '\\'
    findtype='.raw'
    word='i+d'
    #word=choose_type
    for folder, subfolder, files in os.walk(path):
        # print(files)
        #print('current in folder=> '+folder)

        if 'output' not in folder and '_internal' not in folder:
            print('current in folder=> ' + folder)
            for file in files:
                if findtype in file:

                    filenames=folder+'\\'+file
                    if word=='d':
                        if 'depth' in file.lower():
                            imagelist.append(filenames)

                    if word=='i':
                        if 'ir' in file.lower():
                            imagelist.append(filenames)
                    if word=='i+d':
                        if 'ir' in file.lower() or 'depth' in file.lower():
                            imagelist.append(filenames)
            if imagelist!=[]:
                create_outputfolder(folder)
                process_raw(folder)
                cleansettings()

    #img = np.reshape(img, newshape=(h, w, 2))
    #cv2.imshow('img', img)
    #h, w, _ = img.shape
    #a = img.shape





    #depthimage,confidence=raw16_to_raw12(depth)
    #irimage = irenhance(ir)
   # a = depthimage.shape
    #print('depthimage='+ str(a))

    #img = raw16_to_raw12(img)
    #img = cv2.rotate(img, cv2.ROTATE_180)
    #img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    #print(confidence)
    #imgcopy = img[:,:,0].copy()
    #np.savetxt('depthimage.txt', depthimage)
    #depthimage=depthimage.reshape(h,w,1)
    #confidence = confidence.reshape(h, w, 1)
    #ir = ir.reshape(h, w, 1)
    #print(depthimage)
    #min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(depthimage)
    #mean=np.mean(depthimage)
    #medium=np.median(depthimage)
    #print(max_val)
    #depthimage=255/max_val*depthimage
    #min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(depthimage)
    #print(max_val)
    #_,confidence=cv2.threshold(confidence,0.5,1,cv2.THRESH_BINARY)
    #_, depthimage = cv2.threshold(depthimage, 150, max_val+1, cv2.THRESH_BINARY)

    #depthimage=cv2.resize(depthimage, (int(w), int(h)), interpolation=cv2.INTER_AREA)
    #confidence = cv2.resize(confidence, (int(w), int(h)), interpolation=cv2.INTER_AREA)


    #cv2.imshow('confidence',confidence)
    #cv2.imshow('ROI_image',depthimage)
    #cv2.imshow('ir', ir)



    #cv2.setMouseCallback('ROI_image',on_mouse)
    #cv2.imwrite('temp.jpg', imgcopy)
    #cv2.waitKey(0)