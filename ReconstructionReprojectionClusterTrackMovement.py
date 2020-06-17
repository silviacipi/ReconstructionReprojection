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
import json
from json import JSONEncoder
from json import JSONDecoder


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

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
    #raw_input('normalising...press enter')
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


'''
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
'''
'''
def changeCentre(pathToSavu,newCentre,recAlg):
	f1 = h5py.File(pathToSavu, 'r+')
    	data = f1['entry/plugin/   3 /data']
	print data
	newCentre1='{"in_datasets":[],"init_vol":null,"res_norm":false,"ratio":0.95,"log":false,"algorithm":"'+str(recAlg)+'","out_datasets":[],"centre_pad":false,"outer_pad":true,"log_func":"np.nan_to_num(-np.log(sino))","n_iterations":100,"force_zero":[null, null],"vol_shape":"fixed", "preview":[],"centre_of_rotation":'+str(newCentre)+',"FBP_filter":"ram-lak"}'
    	data[...]=newCentre1
	
    	print 'cedntre changed', data#
	f1.close()
'''
def changeSavuFile(savuFile,dictionary):
    dataFolder='AstraReconGpu'
    contLoop=True
    pathTot=''
    print 'changing file ', savuFile
    mypathTemp=h5py.File(savuFile,'r+') 
    contLoop, pathToData, pathTot=myRecTot(mypathTemp,contLoop,pathTot,dataFolder,False)
    #raw_input('press enter')
    if not (contLoop):
            print 'database "',dataFolder,'" found in  ', pathTot
            pathTot=pathTot+'/data'
            #print type(str(mypathTemp[str(pathTot)].value[0]))
            print type(mypathTemp[str(pathTot)][...])
            encodedNumpyData = json.dumps(mypathTemp[str(pathTot)].value[0], cls=NumpyArrayEncoder)  
            print type(encodedNumpyData)
            decodedArrays = json.loads(encodedNumpyData)
            print type(decodedArrays.encode('ascii','ignore'))
            tmp=decodedArrays.encode('ascii','ignore')
            json_acceptable_string = tmp.replace("'", "\"")  
            print json_acceptable_string, type(json_acceptable_string)
            myDict=json.loads(json_acceptable_string)
            #print type(myDict)
            for key in dictionary:
                print 'changing ', key, myDict[key], dictionary[key]
                myDict[key]=dictionary[key]
            #print myDict['centre_of_rotation']
            #myDict['centre_of_rotation']=140
            print myDict['centre_of_rotation']
            encodedNumpyData=json.dumps(myDict,cls=NumpyArrayEncoder,separators=(',', ':'))
            #print np.array(encodedNumpyData), type(np.array(encodedNumpyData))
            mypathTemp[str(pathTot)][...]=np.array(encodedNumpyData)
            #print type(mypathTemp[str(pathTot)][...])
            #mypathTemp[str(pathTot)][...]=encodedNumpyData
    else:
            print 'failed updating'
    mypathTemp.close()
    print 'file closed'

def myRecTot(obj,continueLoop,pathTot,dataFolder,dataFolderTrue=True):  
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
                    continueLoop,temp,tempPath= myRecTot(obj[name],continueLoop,tempPath,dataFolder,dataFolderTrue)
                else:
                    continue
            else:
                test=obj[name]
		if dataFolderTrue==True:
			temp1='/'+dataFolder
                	if temp1 in test.name:
                    		continueLoop=False
                    		tempPath=pathTot+'/'+name
                    	return continueLoop,test.name,tempPath
		else:
			temp1=dataFolder
			if temp1==test.value[0]: 
                    		tempPath=pathTot
                    		continueLoop=False
                    		return continueLoop,test.name,tempPath
            i=i+1
        if (i-1)>len(obj.items()):
            tempPath=''
    pathTot=pathTot+tempPath
    return continueLoop,temp, pathTot

'''
def myRecSavu(obj,continueLoop,pathTot,dataFolder):  
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
                    continueLoop,temp,tempPath= myRecSavu(obj[name],continueLoop,tempPath,dataFolder)
                else:
                    continue
            else:
                test=obj[name]
                temp1=dataFolder
                if temp1==test.value[0]: 
                    tempPath=pathTot
                    continueLoop=False
                    return continueLoop,test.name,tempPath
            i=i+1
        if (i-1)>len(obj.items()):
            tempPath=''
    pathTot=pathTot+tempPath
    print pathTot
    return continueLoop,temp, pathTot
'''

def fullPath(folder,fileNr,outputDirectory,year=''):
	if year=='':
		now = datetime.datetime.now()
		year=str(now.year)
	else:
		year=str(year)
	directory=''
	if isinstance(fileNr,int):
   		directory='/dls/i13-1/data/'+year+'/'+folder+'/processing/ptychography/tomo/'+str(fileNr)+'/'
	else:
		directory=folder+outputDirectory+'/'
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

def restart(nIter, a,c,b,directory2,fileNr,dataSmall,dataSmallOriginal,pathToSavu,dictionary,directoryResults,totMovement,centre,angleStep):
	for i in range (nIter):
                print 'iteration',i
		if isinstance(fileNr,int):
			name2=directory2+str(fileNr)+'_tomoNX_RegisteredIteration'+str(i)+'.h5'  
		else:
			name2=directory2+fileNr+'_'+str(i)+'.h5'  
                print 'saving in directory', name2   
		#raw_input('check saving directory')
		width=c
        	height=b   
		merlinTomo=h5py.File(name2,"w")        	
        	dsetImage=merlinTomo.create_dataset('data', (a,b,c), 'f')
                dsetKey=merlinTomo.create_dataset('image_key', data=np.zeros(a), dtype='f')  
                dsetImage[...]=dataSmall
                merlinTomo.close() 
		#if dictionary['centre_of_rotation']<0:
		if centre<0:
			
			mycent=findCentre(dataSmall[0,:,:],dataSmall[-1,:,:])
			dictionary['centre_of_rotation']=mycent
			#print 'new centre', mycent
			#raw_input('press enter')
		else:
			#mycent=centre
			mycent=dictionary['centre_of_rotation']
		
		print 'updating savu configuration file'
		changeSavuFile(pathToSavu,dictionary)
		#raw_input('press enter')
		#changeCentre(pathToSavu,mycent,recAlg)
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
    		contLoopCRecon, pathToDataRecon, pathTotRecon=myRecTot(mypathRecon,contLoopRecon,pathTotRecon,dataFolderRecon,True)
		rec=np.nan_to_num(np.array(mypathRecon[str(pathTotRecon)]))
		print 'new reconstruction dimensions',np.shape(rec)
                p1=np.sum(rec,0)
		print 'calculating rotation matrix'
		rows,height,cols=np.shape(rec)
                
		tmp=np.zeros(np.shape(rec))
		pippo=np.copy(dataSmall)
		'''
		if dictionary['algorithm']=='FBP_CUDA':
			mycentNew=int(c/2)
		else:
			mycentNew=mycent

		'''
		mypathRecon.close()
		print mycent
		#raw_input('press enter')
		
		test=rotateRegisterShiftMPI(int(a),np.copy(rec),np.copy(pippo),np.copy(dataSmallOriginal),totMovement,cols, rows, height,angleStep,crop, mycent)

		dataSmall=test

def tomography(folder,fileNr,pathToSavu, dictionary,dataFolder='data',nIter=10,crop=[0,-1,0,-1], angleRange=180.0,normCrop=[0,50,0,50],year='',outputDirectory='test'):
    alpha=1.0
    directory=fullPath(folder,fileNr,outputDirectory,year)
    directory2=''
    print 'directory', directory
    #raw_input('press enter')
    if isinstance(fileNr,int):
    	directory2=directory+outputDirectory+'/'
    else:
	directory2=directory

    if not os.path.exists(directory2):
	os.makedirs(directory2)
    
    pathToSavu=savuFile(pathToSavu,directory2)
    
    print 'path to savu config ', pathToSavu
    #raw_input('press enter')
    if isinstance(fileNr,str):
    	nxsfileName=directory+fileNr+'.h5'
	print 'it is a string'
    else:
    	nxsfileName=directory+str(fileNr)+'_tomoNX.h5'
    print 'file containing projections',nxsfileName
    mypath=h5py.File(nxsfileName,'r') 
    print 'looking for "',dataFolder, '" in the tree...'
    contLoop=True
    pathTot=''
    centre=dictionary['centre_of_rotation']
    #mycent=dictionary['centre_of_rotation']
    #mycent=centre
    print directory, directory2
    #raw_input('press enter')
    contLoop, pathToData, pathTot=myRecTot(mypath,contLoop,pathTot,dataFolder,True)
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
	dataSmallOriginal=np.zeros(npdata.shape)
	print 'normalising projections..'
	if isinstance(fileNr,int):
		print 'it is an int I normalise'
        	dataSmallOriginal=normTomo(npdata,fromX,toX,fromY,toY)
	else:
		print 'it is a string I dont normalise'
		dataSmallOriginal=npdata#normTomo(npdata,fromX,toX,fromY,toY)
	#plt.imshow(dataSmallOriginal[0,:,:])
	#raw_input('press enter')
	#plt.show()
	print 'projections normalised'
        print a,b,c, ' file images to analyse' 
        angleStep=angleRange/float((a-1))
	print 'angleStep',angleStep
        xyCorrection=np.zeros((2,a))
	
	dataSmall=dataSmallOriginal.copy()
	directoryResults=''
	restart(nIter,a, c,b,directory2,fileNr,dataSmall,dataSmallOriginal,pathToSavu,dictionary,directoryResults,totMovement,centre,angleStep)
	'''        
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
		#if dictionary['centre_of_rotation']<0:
		if centre<0:
			
			mycent=findCentre(dataSmall[0,:,:],dataSmall[-1,:,:])
			dictionary['centre_of_rotation']=mycent
			#print 'new centre', mycent
			#raw_input('press enter')
		else:
			#mycent=centre
			mycent=dictionary['centre_of_rotation']
		
		print 'updating savu configuration file'
		changeSavuFile(pathToSavu,dictionary)
		#raw_input('press enter')
		#changeCentre(pathToSavu,mycent,recAlg)
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
    		contLoopCRecon, pathToDataRecon, pathTotRecon=myRecTot(mypathRecon,contLoopRecon,pathTotRecon,dataFolderRecon,True)
		rec=np.nan_to_num(np.array(mypathRecon[str(pathTotRecon)]))
		print 'new reconstruction dimensions',np.shape(rec)
                p1=np.sum(rec,0)
		print 'calculating rotation matrix'
		rows,height,cols=np.shape(rec)
                
		tmp=np.zeros(np.shape(rec))
		pippo=np.copy(dataSmall)

		#if dictionary['algorithm']=='FBP_CUDA':
		#		mycentNew=int(c/2)
		#else:
		#	mycentNew=mycent

		mypathRecon.close()
		print mycent
		#raw_input('press enter')
		test=rotateRegisterShiftMPI(int(a),np.copy(rec),np.copy(pippo),np.copy(dataSmallOriginal),totMovement,cols, rows, height,angleStep,crop, mycent)

		dataSmall=test
	'''
    else:
        print 'database "', dataFolder,'" not found!'
    mypath.close()


#########For testing function
if __name__ == "__main__":


    	
	'''
	To START from scratch
	folder='mg22189-1'
	fileNr=250366
	or to RESTART a registration that stopped
	folder='/dls/i13-1/data/2019/mg22189-1/processing/ptychography/tomo/250366/'
	fileNr='250366_tomoNX_RegisteredIteration1'
	'''
	year=2019
	#folder='mg22189-1'
	#fileNr=250366
	folder='/dls/i13-1/data/2019/mg22189-1/processing/ptychography/tomo/250366/'
	fileNr='250366_tomoNX_RegisteredIteration0_37'
	nIter=100
	#centre=156#-1
	minSlice=10
	maxSlice=100#290
	minCol=200
	maxCol=700
	crop=[minSlice,maxSlice,minCol,maxCol]
	#normFromX=0
	#normToX=50
	normFromY=0
	normToY=50
	normFromX=0
	normToX=50   
	normCrop=[normFromX,normToX,normFromY,normToY]
	angleRange=140.0
	pathToSavu='/dls/i13-1/data/2019/mg22189-1/processing/savuCOR2.nxs'
	counter=14
	outputDirectory='Registration_cor400_SIRT_testSC2'
	#recAlg='FBP_CUDA'
        dictionary={'centre_of_rotation': 400, 'algorithm': 'SIRT_CUDA'}
	tomography(folder,fileNr,pathToSavu,dictionary,'data',nIter, crop,angleRange,normCrop,year,outputDirectory)
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




    
