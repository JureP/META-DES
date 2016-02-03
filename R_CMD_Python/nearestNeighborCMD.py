__author__ = 'Podlogar'
import numpy as np
from sklearn.neighbors import NearestNeighbors
import csv
import sys
import ast


def putCSVtoList(file):
    ''' prebere csv iu file in ga zapise v list '''
    vektor = []
    with open(file) as beri:
        vrstica  = csv.reader(beri, delimiter=',')
        for element in vrstica:
            #vektor.append(element[0])
            vektor.append(element[0])
    return vektor


Xsosedi = np.genfromtxt('C:/Users/Podlogar/Documents/DES_ektimo/Fin/SosediPython/sosedSet.csv', delimiter=',')
Xmeta = np.genfromtxt('C:/Users/Podlogar/Documents/DES_ektimo/Fin/SosediPython/metaSet.csv', delimiter=',')
Xtest = np.genfromtxt('C:/Users/Podlogar/Documents/DES_ektimo/Fin/SosediPython/testSet.csv', delimiter=',')

OPsosedi = np.genfromtxt('C:/Users/Podlogar/Documents/DES_ektimo/Fin/SosediPython/OP_sosedSet.csv', delimiter=',')
OPmeta = np.genfromtxt('C:/Users/Podlogar/Documents/DES_ektimo/Fin/SosediPython/OP_metaSet.csv', delimiter=',')
OPtest = np.genfromtxt('C:/Users/Podlogar/Documents/DES_ektimo/Fin/SosediPython/OP_testSet.csv', delimiter=',')


shraniKje = "C:/Users/Podlogar/Documents/DES_ektimo/Fin/SosediPython/"

def najdiSosede(nSosed, metric, shraniKot, Xsosedi, Xmeta):
    # metric = putCSVtoList('metric.csv')
    for K in nSosed:
        print(K)
        nearestNeighborRegion = NearestNeighbors(n_neighbors = K, metric = metric)
        nearestNeighborRegion.fit(Xsosedi)
        sosed1 = nearestNeighborRegion.kneighbors(Xmeta)[1]
        sosediFM = sosed1 + 1
        ime = shraniKot.format(K, metric)
        print(ime)
        np.savetxt(ime, sosediFM, delimiter=",")

		
def shraniSosede(stSosedov):
	stSosedov = ast.literal_eval(stSosedov)
	shraniIme = 'matrikaSosedje[trainBL]trainBL[sosedSet]sosedSet[metaSet]metaSet[K]{}[knnALG]{}.csv'
	shraniKotUcenje = shraniKje + shraniIme
	najdiSosede(stSosedov, 'chebyshev', shraniKotUcenje, Xsosedi, Xmeta)
	shraniImeOP = 'matrikaSosedjeOP[trainBL]trainBL[sosedSet]sosedSet[metaSet]metaSet[K]{}[knnALG]{}.csv'
	shraniKotUcenjeOP = shraniKje + shraniImeOP
	najdiSosede(stSosedov, 'chebyshev', shraniKotUcenjeOP, OPsosedi, OPmeta)
	#najdiSosede([350], 'l2', shraniKotUcenje, Xsosedi, Xmeta)
	
	
	shraniImeNapovedovanje = 'matrikaSosedje[trainBL]trainBL[sosedSet]metaSet[metaSet]testSet[K]{}[knnALG]{}.csv'
	shraniKotNapovedovanje = shraniKje + shraniImeNapovedovanje
	najdiSosede(stSosedov, 'chebyshev', shraniKotNapovedovanje, Xmeta, Xtest)
	shraniImeNapovedovanjeOP = 'matrikaSosedjeOP[trainBL]trainBL[sosedSet]metaSet[metaSet]testSet[K]{}[knnALG]{}.csv'
	shraniKotNapovedovanjeOP = shraniKje + shraniImeNapovedovanjeOP
	najdiSosede(stSosedov, 'chebyshev', shraniKotNapovedovanjeOP, OPmeta, OPtest)
	#najdiSosede([350], 'l2', shraniKotNapovedovanje, Xmeta, Xtest)


if __name__ == "__main__":
	shraniSosede(sys.argv[1])

