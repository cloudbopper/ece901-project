import matplotlib.pyplot as plt

# Training Error versus the Number of Epoch for parallel and serial case
def TrainErrVsEpoch(xlist,ylist):
    assert len(xlist) == len(ylist)
    for pIn in range(len(xlist)):
        labelp = 'p = '+ str(pIn+1)		
        plt.plot(xlist[pIn],ylist[pIn] ,label=labelp)
    
    #plt.axis([0, 6, 0, 20])
    plt.ylabel('Training Loss')
    plt.xlabel('Number of Epochs')	
    plt.legend(loc='upper right')
    plt.title('Training loss versus Number of epochs with Averaging across threads')
    plt.show()    

	
#TODO1: Another plot based on definition of speedup with respect serial case? (Speed up versus number of cores)	
#TODO2: Plot comparing time per epoch versus number of threads?	
	


