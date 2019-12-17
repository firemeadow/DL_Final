import numpy as np
import sklearn.decomposition as dc
import sklearn.preprocessing as pp

# fileName  self explainatory
def loadData(fileName):
    return np.loadtxt(fileName, delimiter = ',')

# data      sample x feature matrix
# bufSz     size of eigencombing buffer
# gap       space between samples
# pDex      index of the current price in the feature vector
def genEig(data, bufSz, gap, pDex):
    assert data.shape[1] <= (bufSz / gap) #buffer too small, the eig dimensionality will line up to the sample size, not the number of features

    outEig = []
    outCst = []

    #start after the buffer, don't include the last point
    numPoints = data.shape[0] - (bufSz + gap)
    eigPoints = int(np.floor(bufSz / gap))

    print(numPoints)
    print(eigPoints)

    for i in range(numPoints):
        #temp matrix to store points for this eigenvector
        tempMat = []
        
        #create a list to store the sequence for this eigenvector
        outCst.append([])
        
        for j in range(eigPoints):
            tempMat.append(data[i + (j * gap)])
            outCst[i].append(data[i + (j * gap)][pDex])

        #append the real price to the end of the sequence
        print("adding real datapoint: ", bufSz + i + gap, "/", data.shape[0] - 1)
        outCst[i].append(data[bufSz + i + gap][pDex])

        #calculate the eigenvector over the gapped points
        print("calculating primary component: ", i)
        pFunc = dc.PCA(n_components = 1)
        eVec = pFunc.fit_transform(pp.normalize(tempMat, axis = 0))
        
        outEig.append(eVec)

    return outEig, outCst
