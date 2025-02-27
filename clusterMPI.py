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
import mpi4py.MPI
from multiprocessing import Process, Pipe,Array, Pool, Queue
import multiprocessing as mp
from contextlib import closing



import threading
import time

threadLock = threading.Lock()
threads = []
maximumNumberOfThreads=5
threadLimiter = threading.BoundedSemaphore(maximumNumberOfThreads)

class myThread (threading.Thread):
    def __init__(self, number, array):
        threading.Thread.__init__(self)
	#print 'current thread id ',threading.current_thread().ident
        #self.threadID = threadID
        self.number = number
        self.array=array
	self.arrayOut=np.zeros(np.shape(array))
    def run(self):
	threadLimiter.acquire()
        #print "Starting " + self.name
#         # Get lock to synchronize threads
	#print self.number
	try:
        	p1,p2=test2(self.number, self.array[self.number,:,:])
		#print 'current thread acquire ',threading.current_thread().ident
        	threadLock.acquire()
		self.array[p1,:,:]=p2
		#print self.arrayOut[p1,:,:]
        	threadLock.release()
	finally:
            	threadLimiter.release()



def test2(number, data):
	data2=np.ones(np.shape(data))*number
	return number,data2

def test(nI):
        data=np.zeros((nI,2,2))
	nMaxThreads=5
	nthreads=0
	for i in range(nI):
		thread = myThread( i, data)
		thread.start()
		
        	threads.append(thread)
        	#imageAnalysis(img,threshold,myIter,fileName)        
        for t in threads:
            	t.join()
	#for t in threads:
	#	print t.arrayOut
	
        
        print "Exiting Main Thread"
	print data


class registrationThread (threading.Thread):
    def __init__(self, indexPerProc,processNr,rec,dataSmall,cols, rows, height,angleStep,crop, mycent):
        threading.Thread.__init__(self)
	#print 'current thread id ',threading.current_thread().ident
        #self.threadID = threadID
	self.dataSmall=dataSmall
	self.indexPerProc=indexPerProc
	self.rec=rec
	self.cols=cols
	self.rows=rows
	self.height=height
	self.angleStep=angleStep
	self.crop=crop
	self.mycent=mycent
    def run(self):
	threadLimiter.acquire()
	try:
        	p1,p2=rotateRegShiftLoop(self.indexPerProc,self.rec,self.dataSmall,self.cols, self.rows, self.height,self.angleStep,self.crop, self.mycent)
        	threadLock.acquire()
		self.dataSmall[p1,:,:]=p2
        	threadLock.release()
	finally:
            	threadLimiter.release()

def rotateRegShiftLoop(indexPerProc,rec,dataSmall,cols, rows, height,angleStep,crop, mycent):
	alpha=1
	for j in indexPerProc:
		index,dst1,p1=rotateRegisterShift(rec.copy(),dataSmall[j,:,:],cols, rows, height,angleStep, j ,crop, mycent)
		dataSmall[index,:,:]=alpha*dst1+(1-alpha)*p1
	return indexPerProc, dataSmall[indexPerProc,:,:]

def rotateMatrix(rec,M,cols,rows,height):
    for l in range (height):
		dst = cv2.warpAffine(rec[:,l,:],M,(cols,rows))
		rec[:,l,:]=dst
    return rec


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
	#plt.figure(1)
	#plt.imshow(p1)
        sliceFrom=crop[0]
        sliceTo=crop[1]
	colFrom=crop[2]
	colTo=crop[3]
	print 'register...'
        shift, error, diffphase = register_translation(p[sliceFrom:sliceTo,colFrom:colTo], p1[sliceFrom:sliceTo,colFrom:colTo], 100)
        p=dataSmallj
	#plt.figure(2)
	#plt.imshow(p)
        #print shift, error, diffphase
        move=shift[1]
        print shift, error, diffphase, move
	M1 = np.float32([[1,0,-move],[0,1,-shift[0]]])
        print 'matrix calculated'
        dst1 = cv2.warpAffine(p,M1,(e,d))
	#plt.figure(3)
	#plt.imshow(dst1)
	#plt.show()
	return j, dst1,p1


def rotateRegisterShiftMPI(a,rec,dataSmall,cols, rows, height,angleStep, crop, mycent):
	print 
    	#size=mp.cpu_count()
	size=5
    	numImPerProc= a/size#
	result=np.zeros(np.shape(dataSmall))
    
    	for processNr in range(size):#(size):
		b= a if  processNr==(size-1) else ((processNr+1)*numImPerProc+1)
		indexPerProc=[j for j in range((processNr*numImPerProc+1),b)]
	
		print 'sending thread ', processNr
		thread = registrationThread(indexPerProc,processNr,rec.copy(),dataSmall,cols, rows, height,angleStep,crop, mycent)
		thread.start()
        	threads.append(thread)

        for t in threads:
            	t.join()
  
	result=dataSmall
        return result









#########For testing function
if __name__ == "__main__":
    	#a=sys.argv[0]
    	#rec=sys.argv[1]
    	#dataSmall=sys.argv[2]
   	#rows=sys.argv[3]
    	#height=sys.argv[4]
   	#angleStep=sys.argv[5]
   	#j=sys.argv[6]
    	#crop=sys.argv[7]
    	#mycent=sys.argv[8]
	test(100)

	#pippo=rotateRegisterShiftMPI(a,rec,dataSmall,cols, rows, height,angleStep, j ,crop, mycent)
