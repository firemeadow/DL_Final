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
def genEig(data, bufSz, gap, pDex, s = True, v = False):
    print("working!")
    if (data.shape[1] >= (bufSz / gap)):
        print("your buffer is a bit small")

    outEig = []
    outCst = []
    outMot = []

    #start after the buffer, don't include the last point
    numPoints = data.shape[0] - (bufSz * gap) - 1
    eigPoints = int(np.floor(bufSz / gap))

    print("points:              ", numPoints)
    print("samples per eig:     ", eigPoints)

    for i in range(numPoints):
        #temp matrix to store points for this eigenvector
        tempMat = []
        #create a list to store the sequence for this eigenvector
        outCst.append([data[bufSz + i + gap][pDex]])
        outMot.append([(data[bufSz + i + gap] - data[bufSz + i]) / gap])
        
        for j in range(eigPoints):
            tempMat.append(data[i + (j * gap)])

        #append the real price to the end of the sequence
        if s or v: print("adding real datapoint: ", bufSz + i + gap, "/", numPoints - 1)
        outCst[i].append(data[bufSz + i + gap][pDex])

        #calculate the eigenvector over the gapped points
        if s or v: print("calculating primary component: ", i)
        pFunc = dc.PCA(n_components = 1)
        eVec = pFunc.fit_transform(pp.normalize(np.asarray(tempMat), axis = 0).T)
        if v: print(eVec)
        outEig.append(eVec)

    return outEig, outCst, outMot

#test code
#a, b, c = genEig(loadData('data.txt'), 256, 2, 5, False)
