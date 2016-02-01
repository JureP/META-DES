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


Xsosedi = np.genfromtxt('C:/Users/Podlogar/Documents/DES_ektimo/python_kNN/Podatki/sosedSet.csv', delimiter=',')
Xmeta = np.genfromtxt('C:/Users/Podlogar/Documents/DES_ektimo/python_kNN/Podatki/metaSet.csv', delimiter=',')


skraniKot = "C:/Users/Podlogar/Documents/DES_ektimo/python_kNN/Sosedi/sosediFM_[K]{}_[metric]{}.csv"

def najdiSosede(nSosed, shraniKot):
	nSosed = ast.literal_eval(nSosed)
	# metric = putCSVtoList('metric.csv')
	metric = 'chebyshev'
	for K in nSosed:
		print(K)
		nearestNeighborRegion = NearestNeighbors(n_neighbors = K, metric = metric)
		nearestNeighborRegion.fit(Xsosedi)
		sosed1 = nearestNeighborRegion.kneighbors(Xmeta)[1]
		sosediFM = sosed1 + 1
		ime = shraniKot.format(K, metric)
		print(ime)
		np.savetxt(ime, sosediFM, delimiter=",")


		
def najdiSosede1(nSosed, shraniKot):
	nSosed = ast.literal_eval(nSosed)
	print(shraniKot)
	

if __name__ == "__main__":
	najdiSosede(sys.argv[1], sys.argv[2])
	
# run in cmd:
# python najdiSosede.py [1, 2] C:\<KAMOR NAJ SHRANI>_[K]{}_[metric]{}.csv

# from R through CMD
# system("C:\Python34\python.exe C:\<lokacija>\najdiSosede.py [1, 2] C:\<KAMOR NAJ SHRANI>_[K]{}_[metric]{}.csv")
	

