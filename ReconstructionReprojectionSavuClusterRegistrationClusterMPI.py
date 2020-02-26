import h5py
import tomopy
from tomopy import recon
import numpy as np
from matplotlib import pyplot as plt
from scipy import math
from skimage import data
from skimage.feature import register_translation
from skimage.feature.register_translation import _upsampled_dft
from scipy.ndimage import fourier_shift
import cv2

from scipy import ndimage
import os
import glob
import datetime
import time
from clusterMPI import rotateRegisterShiftMPI
#from clusterMPI import test
from multiprocessing import Process
#import mpi4py.MPI

def findCentre(firstImage,lastImage):
	firstImageFlipped = np.fliplr(firstImage)
	b,c=np.shape(firstImage)
	shift, error, diffphase = register_translation(firstImageFlipped,lastImage, 100)
	mycent=c/2-shift[1]/2
	return mycent


def normProjection(img,fromX,toX,fromY,toY,mean):
	newMean=np.mean(img[fromY:toY,fromX:toX])
	diff=mean-newMean
	imgNorm=img+np.ones(np.shape(img))*diff	
	return imgNorm


def normTomo(npdata,fromX,toX,fromY,toY):
    a,b,c=np.shape(npdata)
    img=np.zeros(np.shape(npdata))
    meanValue=np.mean(npdata[fromY:toY,fromX:toX])	
    for idx in range (0,a):
	img[idx,:,:]=normProjection(npdata[idx,:,:],fromX,toX,fromY,toY,meanValue)
    return img   

def smoothData(dataSmall):
    a,b,c=np.shape(dataSmall)
    test=np.zeros(np.shape(dataSmall))
    kernel = np.ones((15,15),np.float32)/25
    for m in range(b):
	test[:,m,:]=cv2.filter2D(dataSmall[:,m,:],-1,kernel)
    return test

def rotateMatrix(rec,M,cols,rows,height):
    for l in range (height):
		dst = cv2.warpAffine(rec[:,l,:],M,(cols,rows))
		rec[:,l,:]=dst
    return rec



def myRec(obj,continueLoop,pathTot,dataFolder):  
    ### recursive function to look for the data database
    temp=None
    i=1
    tempPath=''
    for name, value in obj.items():
        if continueLoop:
            #check if the object is a group
            if isinstance(obj[name], h5py.Group):
                tempPath='/'+name
                if len(obj[name])>0:
                    continueLoop,temp,tempPath= myRec(obj[name],continueLoop,tempPath,dataFolder)
                else:
                    continue
            else:
                test=obj[name]
                temp1='/'+dataFolder
                if temp1 in test.name:
                    continueLoop=False
                    tempPath=pathTot+'/'+name
                    return continueLoop,test.name,tempPath
            i=i+1
        if (i-1)>len(obj.items()):
            tempPath=''
    pathTot=pathTot+tempPath
    return continueLoop,temp, pathTot


def changeCentre(pathToSavu,newCentre):
	f1 = h5py.File(pathToSavu, 'r+')
    	data = f1['entry/plugin/   3 /data']
	print data
	newCentre1='{"in_datasets":[],"init_vol":null,"res_norm":false,"ratio":0.95,"log":false,"algorithm":"SIRT_CUDA","out_datasets":[],"centre_pad":false,"outer_pad":true,"log_func":"np.nan_to_num(-np.log(sino))","n_iterations":100,"force_zero":[null, null],"vol_shape":"fixed", "preview":[],"centre_of_rotation":'+str(newCentre)+',"FBP_filter":"ram-lak"}'
    	data[...]=newCentre1
    	print data


def fullPath(folder,fileNr,year=''):
	if year=='':
		now = datetime.datetime.now()
		year=str(now.year)
	else:
		year=str(year)
   	directory='/dls/i13-1/data/'+year+'/'+folder+'/processing/ptychography/tomo/'+str(fileNr)+'/'
	print 'directory',directory
	return directory

def launchSavu(pathToNxFile,pathToConfigFile,pathToOutDir):

	cmd='sh /dls_sw/i13-1/scripts/Silvia/test2.sh '+pathToNxFile+' '+pathToConfigFile+' '+pathToOutDir
	#cmd='module load savu && savu '+pathToNxFile+' '+pathToConfigFile+' '+pathToOutDir
	try:             
		print 'reconstructing with Savu on cluster'   
		print cmd
		os.system(cmd)
	except:
		print 'command failed'


def follow(thefile):
	thefile.seek(0,2)
	print 'file opened'
	exitLoop=0
	while True:
		line = thefile.readlines()
		if not line:
        		time.sleep(0.1)
        		continue
		for lineComplete in line:
			#print lineComplete.find("Complete")
			if lineComplete.find("Complete")!=-1:
				print line
        			print "file finished"
				exitLoop=1
        			break;
		yield line
		if exitLoop:
			break;

def monitorClusterJob(directory):
	filePath = directory+'user.log'
	print filePath,'filePath'
	logfile = open(filePath,"r")
    	loglines = follow(logfile)
	for line in loglines:
        	print '%s \n' %(line)

def f(task):
	print 'inside Function'
	print 'task', task
'''
def rotateRegisterShift(rec,dataSmallj,cols, rows,height, angleStep, j ,crop,mycent):
	tmp=np.zeros(np.shape(rec))
	if j>0:
		M = cv2.getRotationMatrix2D((cols/2,rows/2),-angleStep*j,1)				
		tmp=rotateMatrix(rec,M,cols,rows,height)
		print 'rotation completed'
        else:
		tmp=rec	
	print 'reproject...'
	p1=np.sum(tmp,0)

	
	M2 = np.float32([[1,0,-(cols/2-mycent)],[0,1,0]])                        
	p=dataSmallj
        d,e=np.shape(p)
	dst1 = cv2.warpAffine(p1,M2,(e,d))
	p1=dst1.copy()
	plt.figure(1)
	plt.imshow(p1)
        sliceFrom=crop[0]
        sliceTo=crop[1]
	colFrom=crop[2]
	colTo=crop[3]
	print 'register...'
        shift, error, diffphase = register_translation(p[sliceFrom:sliceTo,colFrom:colTo], p1[sliceFrom:sliceTo,colFrom:colTo], 100)
        p=dataSmallj
	plt.figure(2)
	plt.imshow(p)
        #print shift, error, diffphase
        move=shift[1]
        print shift, error, diffphase, move
	M1 = np.float32([[1,0,-move],[0,1,-shift[0]]])
        print 'matrix calculated'
        dst1 = cv2.warpAffine(p,M1,(e,d))
	plt.figure(3)
	plt.imshow(dst1)
	plt.show()
	return j, dst1,p1
'''
def tomography(folder,fileNr,pathToSavu, dataFolder='data',centre=-1,nIter=10,crop=[0,-1,0,-1], angleRange=180.0,normCrop=[0,50,0,50],year=''):
    alpha=1.0
    directory=fullPath(folder,fileNr,year)
    nxsfileName=directory+str(fileNr)+'_tomoNX.h5'
    print 'file containing projections',nxsfileName
    mypath=h5py.File(nxsfileName,'r') 
    print 'looking for "',dataFolder, '" in the tree...'
    contLoop=True
    pathTot=''
    mycent=centre
    contLoop, pathToData, pathTot=myRec(mypath,contLoop,pathTot,dataFolder)
    print pathTot
  
    if not contLoop:
        print 'database "',dataFolder,'" found in  ', pathTot
        npdata=np.array(mypath[str(pathTot)])*(-1)
        print npdata.shape
        a,b,c=npdata.shape 
        dataSmall=np.zeros((int(a),b,c))
        counter=0
        img=np.zeros([a,b,c])
	fromX=normCrop[0]
	toX=normCrop[1]
	fromY=normCrop[2]
	toY=normCrop[3]
	print 'normalising projections..'
        dataSmall=normTomo(npdata,fromX,toX,fromY,toY)
	print 'projections normalised'
        print a,b,c, ' file images to analyse' 
        angleStep=angleRange/(a-1)
        xyCorrection=np.zeros((2,a))
	directory2=directory+'test2/'
	if not os.path.exists(directory2):
		 os.makedirs(directory2)
        for i in range (nIter):
                print 'iteration',i
		name2=directory2+str(fileNr)+'_tomoNX_RegisteredIteration'+str(i)+'.h5'  
                print 'saving in directory', name2   
		width=c
        	height=b   
		merlinTomo=h5py.File(name2,"w")        	
        	dsetImage=merlinTomo.create_dataset('data', (a,b,c), 'f')
                dsetKey=merlinTomo.create_dataset('image_key', data=np.zeros(a), dtype='f')  
                dsetImage[...]=dataSmall
                merlinTomo.close() 
		if centre<0:
			mycent=findCentre(dataSmall[0,:,:],dataSmall[-1,:,:])
		else:
			mycent=centre
		print 'updating centre in savu configuration file with value', mycent
		changeCentre(pathToSavu,mycent)
 		launchSavu(name2,pathToSavu,directory2)
		directoryResults=max(glob.glob(os.path.join(directory2, '*/')), key=os.path.getmtime)
		print directoryResults,'directoryResults'
		time.sleep(5)
		monitorClusterJob(directoryResults)
                nameForRecon=directoryResults+'tomo_p1_astra_recon_gpu.h5'
                mypathRecon=h5py.File(nameForRecon,'r') 
    		dataFolderRecon='data'
    		print 'looking for "',dataFolderRecon, '" in the tree...'
    		contLoopRecon=True
    		pathTotRecon=''
    		contLoopCRecon, pathToDataRecon, pathTotRecon=myRec(mypathRecon,contLoopRecon,pathTotRecon,dataFolderRecon)
		rec=np.nan_to_num(np.array(mypathRecon[str(pathTotRecon)]))
		print 'new reconstruction dimensions',np.shape(rec)
                p1=np.sum(rec,0)
		print 'calculating rotation matrix'
		rows,height,cols=np.shape(rec)
                
		tmp=np.zeros(np.shape(rec))
		pippo=dataSmall.copy()   		
		test=rotateRegisterShiftMPI(int(a),rec.copy(),pippo.copy(),cols, rows, height,angleStep,crop, mycent)
		dataSmall=test
    else:
        print 'database "', dataFolder,'" not found!'
    mypath.close()


#########For testing function
if __name__ == "__main__":

    year=2020
    folder='mg23919-1'
    fileNr=296786

    nIter=50
    centre=-1
    minSlice=50
    maxSlice=200
    minCol=50
    maxCol=350
    crop=[minSlice,maxSlice,minCol,maxCol]
    normFromX=0
    normToX=50
    normFromY=0
    normToY=50   
    normCrop=[normFromX,normToX,normFromY,normToY]
    angleRange=180.0
    pathToSavu='/dls/i13-1/data/2019/mg24277-1/processing/registrationSavu2.nxs'
    tomography(folder,fileNr,pathToSavu,'data',centre,nIter, crop,angleRange,normCrop,year)

    
