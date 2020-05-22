import h5py
import tomopy
from tomopy import recon
import numpy as np
from numpy import copy
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
from clusterMPITrackMovement import rotateRegisterShiftMPI
#from clusterMPI import test
from multiprocessing import Process
#import mpi4py.MPI
from shutil import copyfile

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
	#print diff, newMean,mean,fromY,toY,fromX,toX, np.shape(img)
    	#raw_input('press enter')
	return imgNorm


def normTomo(npdata,fromX,toX,fromY,toY):
    a,b,c=np.shape(npdata)
    img=np.zeros(np.shape(npdata))
    meanValue=np.mean(npdata[0,fromY:toY,fromX:toX])	
    #print meanValue,fromY,toY,fromX,toX,np.shape(npdata)
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
	
    	print 'cedntre changed', data#
	f1.close()


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
		var=os.system(cmd)
		print 'os system output %s' %(var)
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

def savuFile(pathToSavu,directory2):
    try:
	print 'copying locally savu config file'
	print pathToSavu, directory2
	print(os.path.basename(pathToSavu))
	
	cmd="cp %s %s" %(pathToSavu,directory2)
	print cmd
	#raw_input('press enter')
	os.system(cmd)
    	#shutil.copyfile(pathToSavu, directory)	
    except:
	print 'failed copying savu file'
    newPathToSavu=directory2+os.path.basename(pathToSavu)
    print 'newPathToSavu',newPathToSavu
    #raw_input('press enter')
    return newPathToSavu

def tomography(folder,fileNr,pathToSavu, dataFolder='data',centre=-1,nIter=10,crop=[0,-1,0,-1], angleRange=180.0,normCrop=[0,50,0,50],year='',outputDirectory='test'):
    alpha=1.0
    directory=fullPath(folder,fileNr,year)
    directory2=directory+outputDirectory+'/'
    if not os.path.exists(directory2):
	os.makedirs(directory2)
    
    pathToSavu=savuFile(pathToSavu,directory2)
    
    print 'path to savu config ', pathToSavu
    #raw_input('press enter')
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
	totMovement=np.zeros((int(a),2))
        counter=0
        img=np.zeros([a,b,c])
	fromX=normCrop[0]
	toX=normCrop[1]
	fromY=normCrop[2]
	toY=normCrop[3]
	print 'normalising projections..'
        dataSmallOriginal=normTomo(npdata,fromX,toX,fromY,toY)
	#plt.imshow(dataSmallOriginal[0,:,:])
	#plt.show()
	print 'projections normalised'
        print a,b,c, ' file images to analyse' 
        angleStep=angleRange/float((a-1))
	print 'angleStep',angleStep
        xyCorrection=np.zeros((2,a))
	
	dataSmall=dataSmallOriginal.copy()
	directoryResults=''
        for i in range (nIter):
		if i==26:
			print 'waiting 60 seconds for enabling savu again on cluster...'
			time.sleep(60)
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
		attemptLaunching=0
		while attemptLaunching<5:
			if directoryResults!= max(glob.glob(os.path.join(directory2, '*/')), key=os.path.getmtime):
				directoryResults=max(glob.glob(os.path.join(directory2, '*/')), key=os.path.getmtime)
				break
			else:
				attemptLaunching+=1
				time.sleep(1.0)
				print 'waiting to launch savu, attempt %d of 5' %(attemptLaunching)
		if attemptLaunching==5:
			print 'savu not launched...'
			break
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
		pippo=np.copy(dataSmall)
		test=rotateRegisterShiftMPI(int(a),np.copy(rec),np.copy(pippo),np.copy(dataSmallOriginal),totMovement,cols, rows, height,angleStep,crop, mycent)

		dataSmall=test
    else:
        print 'database "', dataFolder,'" not found!'
    mypath.close()


#########For testing function
if __name__ == "__main__":


    	year=2019
	#folder='mg23919-1'
	folder='cm22975-4'
	fileNr=285448

	nIter=1
	centre=140#-1
	minSlice=10
	maxSlice=100#290
	minCol=50
	maxCol=300
	crop=[minSlice,maxSlice,minCol,maxCol]
	#normFromX=0
	#normToX=50
	normFromY=0
	normToY=50
	normFromX=350
	normToX=390   
	normCrop=[normFromX,normToX,normFromY,normToY]
	angleRange=180.0
	pathToSavu='/dls_sw/i13-1/scripts/Silvia/Reprojection/ReconstructionReprojection/savuFBP.nxs'
	counter=14
	outputDirectory='testFBP'
	tomography(folder,fileNr,pathToSavu,'data',centre,nIter, crop,angleRange,normCrop,year,outputDirectory)
	'''
	for j in range(135,145,1):
		centre=j
		print 'new centre of rotation', centre
		outputDirectory='test%d' %(counter)#within the fileNr folder
		#for i in range(145,165,5):
		#centre=i
		tomography(folder,fileNr,pathToSavu,'data',centre,nIter, crop,angleRange,normCrop,year,outputDirectory)
		counter=counter+1
	'''




    
