## META-DES financni podatki
set.seed(10)
OkoljePodatki <- 'C:/Users/Podlogar/Documents/DES_ektimo/Fin/Podatki_proxySmall'
OkoljeIzvedba <- 'C:/Users/Podlogar/Documents/DES_ektimo/Fin/'
OkoljeFunkcije <- paste0(OkoljeIzvedba,'/Funkcije')

OkoljeRezultat <- paste0(OkoljeIzvedba, '/Rezultati')
OkoljeRezultatCV <- paste0(OkoljeRezultat)
######################################################################
OkoljeMetaPrblm <- paste0(OkoljeRezultatCV, '/Meta Problem')
OkoljeSosediMP <- paste0(OkoljeMetaPrblm, '/matrikaSosedi')
OkoljeMetaPrblm_VALID <- paste0(OkoljeRezultatCV, '/Meta Problem (VALID)')
OkoljeSosedi_VALID <- paste0(OkoljeMetaPrblm_VALID, '/matrikaSosedi')

dir.create(OkoljeSosediMP)
dir.create(OkoljeSosedi_VALID)

## imena mnozic za potrebe poimenovanja.
imenaMnozic <- c('trainBL', 'sosedSet', 'metaSet', 'testSet')
imenaMnozicTest <- imenaMnozic[c(1,3,4)]


{## nastavitve parametrov
MetaMode = c('individual')
K = 350 ## stevilo sosedov za region of competence (vektor)
K = c(2,3)
knnALG = "chebyshev" ## algoritem po katerem get.knnx isce sosede
metaALG = c('rf') ## klasifikacijski algoritmi uporabljeni pri ucenju meta klasifikatorja
hC = c(0.7)
hC_ONE = c(0.7)
kompetThrsh = c(0.7)
}
{## parametri za zagon pythona
pythonEXE <- "C:/Python34/python.exe "
pythonOkolje <- "C:/Users/Podlogar/Documents/DES_ektimo/Fin/SosediPython/"
pythonFUN <-  "nearestNeighborCMD.py"
parameter <- paste0('[', paste0(K, collapse=","), ']')
}


baseLearner <- c('rf_zs', 'svm_mg', 'rf_el')

	
{## knjiznice in funkcije
setwd(OkoljeFunkcije)
source("metaProblemSAVE_funkcija.r")
source("probToClass_funkcija.r")
source("ucenjeMetaKlasifikator_funkcija.r")
source("napovedEnsemble_funkcija.r")
source("optMETA_DES_CARET_funkcija.r")
source("ucenjeMetaKlasifikatorONE_funkcija.r")

library(mlbench)
library(MASS)
library(caret)
library(kernlab)
}

{## podatki
setwd(OkoljePodatki)
	sosedSet <- readRDS('sosedSet.rds')
	ySosedSet <- sosedSet$response
	sosedSet <- sosedSet[,3:174]
	
	metaSet <- readRDS('metaSet.rds')
	yMetaSet <- metaSet$response
	metaSet <- metaSet[,3:174]
	
	testSet <- readRDS('validSet.rds')
	yTestSet <- testSet$response
	testSet <- testSet[,3:174]
}

## NAPOVEDI BASE LEARNERJEV ###############################################	###########################################################################		
	setwd(OkoljePodatki)
	OP_sosedi <- readRDS('OP_sosedi.rds')[,-c(1,2)]
	OP_meta <- readRDS('OP_meta.rds')[,-c(1,2)]
	OP_test <- readRDS('OP_valid.rds')[,-c(1,2)]

	
sosedSet <- sosedSet[1:5000, ]
ySosedSet <- ySosedSet[1:5000]
metaSet <- metaSet[1:5000, ]
yMetaSet <- yMetaSet[1:5000]
testSet <- testSet[1:5000, ]
yTestSet <- yTestSet[1:5000]

OP_sosedi <- OP_sosedi[1:5000, ]
OP_meta <- OP_meta[1:5000, ]
OP_test <- OP_test[1:5000, ]

## izracun knn chebyshev v python

## shrani kot csv za izracun knn v metriki chebyshev
setwd(pythonOkolje)
write.table(metaSet, file = 'metaSet.csv', sep = ',', col.names = FALSE, row.names = FALSE)
write.table(sosedSet, file = 'sosedSet.csv', sep = ',', col.names = FALSE, row.names = FALSE)
write.table(testSet, file = 'testSet.csv', sep = ',', col.names = FALSE, row.names = FALSE)

write.table(OP_meta, file = 'OP_metaSet.csv', sep = ',', col.names = FALSE, row.names = FALSE)
write.table(OP_sosedi, file = 'OP_sosedSet.csv', sep = ',', col.names = FALSE, row.names = FALSE)
write.table(OP_test, file = 'OP_testSet.csv', sep = ',', col.names = FALSE, row.names = FALSE)

## zazeni nearestNeighborCMD.py
cmdUkaz <- paste(pythonEXE,paste0(pythonOkolje, pythonFUN), parameter)
system(cmdUkaz)
system("C:/Python34/python.exe C:/Users/Podlogar/Documents/DES_ektimo/Fin/SosediPython/nearestNeighborCMD.py [1,2]")


## shranjevanje sosedov za ucenje meta klasifikatorja
for(nSosedi in K){
	for(alg in knnALG){
		setwd(pythonOkolje)
		## branje sosedov region of competence [ucenje]
		imeSosedi <- paste0('matrikaSosedje[trainBL]',imenaMnozic[1] ,'[sosedSet]',imenaMnozic[2], 
										'[metaSet]', imenaMnozic[3], '[K]',nSosedi, '[knnALG]', alg)
		sosediRC <- read.csv(paste0(imeSosedi, '.csv'), header = FALSE)
		sosediRC <- apply(sosediRC, 2, as.vector)
		## shranjevanje sosedov region of competence [ucenje]
		setwd(OkoljeSosediMP)
		saveRDS(sosediRC, paste0(imeSosedi, '.rds'))
		
		## branje sosedov output proifile [ucenje]
		setwd(pythonOkolje)
		imeSosediOP <- paste0('matrikaSosedjeOP[trainBL]',imenaMnozic[1] ,'[sosedSet]',imenaMnozic[2], 
										'[metaSet]', imenaMnozic[3], '[K]',nSosedi, '[knnALG]', alg)
		sosediOP <- read.csv(paste0(imeSosediOP, '.csv'), header = FALSE)
		sosediOP <- apply(sosediOP, 2, as.vector)
		## shranjevanje sosedov output proifile [ucenje]
		imeSosediOP <- paste0('matrikaSosedje_OP[trainBL]',imenaMnozic[1] ,'[sosedSet]',imenaMnozic[2], 
										'[metaSet]', imenaMnozic[3], '[Kp]',nSosedi, '[knnALG]', alg)
		setwd(OkoljeSosediMP)
		saveRDS(sosediOP, paste0(imeSosediOP, '.rds'))
	}
}



## shranjevanje sosedov napovedovanje kompetenstnosti
for(nSosedi in K){
	for(alg in knnALG){
		setwd(pythonOkolje)
		## branje sosedov region of competence [ucenje]
		imeSosedi <- paste0('matrikaSosedje[trainBL]',imenaMnozic[1] ,'[sosedSet]',imenaMnozic[3], 
										'[metaSet]', imenaMnozic[4], '[K]',nSosedi, '[knnALG]', alg)
		sosediRC <- read.csv(paste0(imeSosedi, '.csv'), header = FALSE)
		sosediRC <- apply(sosediRC, 2, as.vector)
		## shranjevanje sosedov region of competence [ucenje]
		setwd(OkoljeSosedi_VALID)
		saveRDS(sosediRC, paste0(imeSosedi, '.rds'))
		
		## branje sosedov output proifile [ucenje]
		setwd(pythonOkolje)
		imeSosediOP <- paste0('matrikaSosedjeOP[trainBL]',imenaMnozic[1] ,'[sosedSet]',imenaMnozic[3], 
										'[metaSet]', imenaMnozic[4], '[K]',nSosedi, '[knnALG]', alg)
		sosediOP <- read.csv(paste0(imeSosediOP, '.csv'), header = FALSE)
		sosediOP <- apply(sosediOP, 2, as.vector)
		## shranjevanje sosedov output proifile [ucenje]
		imeSosediOP <- paste0('matrikaSosedje_OP[trainBL]',imenaMnozic[1] ,'[sosedSet]',imenaMnozic[3], 
										'[metaSet]', imenaMnozic[4], '[Kp]',nSosedi, '[knnALG]', alg)
		setwd(OkoljeSosedi_VALID)
		saveRDS(sosediOP, paste0(imeSosediOP, '.rds'))
	}
}

