## P2 problem: UPORABA DS (KNORA??)
set.seed(10)

OkoljePodatki <- 'C:/Users/Podlogar/Documents/DES/P2/Podatki'

OkoljeIzvedba <- 'C:/Users/Podlogar/Documents/DES/P2'
OkoljeFunkcije <- paste0(OkoljeIzvedba,'/Funkcije')
imenaMnozic <- c('trainBL', 'sosedSet', 'metaSet', 'testSet')
imenaMnozicTest <- imenaMnozic[c(1,3,4)]

MetaMode = c('individual', 'one')
K = c(5, 7, 10) ## stevilo sosedov za region of competence (vektor)
Kp = 5 ## stevilo sosedov iz output profile (vektor)
knnALG = "kd_tree" ## algoritem po katerem get.knnx isce sosede
metaALG = 'rf' ## klasifikacijski algoritmi uporabljeni pri ucenju meta klasifikatorja
hC = c(0.7, 1)
hC_ONE = c(0.5, 0.7, 1)
kompetThrsh = c(0.7, 0.9)

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

## podatki
setwd(OkoljePodatki)
	trainBL <- readRDS('trainBL.rds')
	yTrainBL <- readRDS('yTrainBL.rds')
	
	sosedSet <- readRDS('sosedSet.rds')
	ySosedSet <- readRDS('ySosedSet.rds')
	
	## ena mnozica ni potrebna za ucenje meta klasifkatorja
	# trainBL <- rbind(trainBL, sosedSet)
	# yTrainSet <- factor(c(yTrainBL, ySosedSet))
	
	metaSet <- readRDS('metaSet.rds')
	yMetaSet <- readRDS('yMetaSet.rds')
	
	testSet <- readRDS('testSet.rds')
	yTestSet <- readRDS('yTestSet.rds')
	
	
	
	## mape v kater se shranjujejo rezultati
	OkoljeRezultat <- paste0(OkoljeIzvedba, '/Rezultati_OA')
	OkoljeRezultatCV <- paste0(OkoljeRezultat)
	######################################################################
	OkoljeBaseLearner <- paste0(OkoljeRezultatCV, '/Base Learner')
	OkoljeRezultatOutputProfile <- paste0(OkoljeRezultatCV, '/Napovedi BL')
	OkoljeMetaPrblm <- paste0(OkoljeRezultatCV, '/Meta Problem')
	OkoljeMetaKlasifikator <- paste0(OkoljeRezultatCV, '/Meta Klasifikator')
	OkoljeMetaPrblm_VALID <- paste0(OkoljeRezultatCV, '/Meta Problem (VALID)')
	OkoljeKompetentnost <- paste0(OkoljeRezultatCV, '/Matrika kompetentnosti')
	OkoljeNapovedEnsemble <- paste0(OkoljeRezultatCV, '/Napoved ensembla')
	OkoljeAccuracy <- paste0(OkoljeRezultatCV, '/Accuracy')
	OkoljeAccuracyBL <- paste0(OkoljeRezultatCV, '/AccuracyBL')
	##
	dir.create(OkoljeRezultat)
	dir.create(OkoljeRezultatCV)
	dir.create(OkoljeBaseLearner)
	dir.create(OkoljeRezultatOutputProfile)
	dir.create(OkoljeMetaPrblm)
	dir.create(OkoljeMetaKlasifikator)
	dir.create(OkoljeMetaPrblm_VALID)
	dir.create(OkoljeKompetentnost)
	dir.create(OkoljeNapovedEnsemble)
	dir.create(OkoljeAccuracy)
	dir.create(OkoljeAccuracyBL)
	
	
	
	## ucenje pool of classifiers
	baseLearner <- NULL
	for(i in 1:10){
	take <- createDataPartition(yTrainBL, p = 0.4, list = FALSE)
	X <- trainBL[take,]
	y <- yTrainBL[take]
	svp <- ksvm(X,y,type="C-svc", prob.model = TRUE)
	# a <- train(X, factor(y), method = 'svmLinear')
	# a$finalModel	
	##plot(svp,data=FM)
		plot(scale(trainBL), col=as.numeric(yTrainBL)*2, pch=as.numeric(yTrainBL), xlab="", ylab="")
		w <- colSums(coef(svp)[[1]] * trainBL[unlist(alphaindex(svp)),])
		b <- b(svp)
		abline(b/w[1],-w[2]/w[1], col = 'black', lwd = 2)
		abline((b+1)/w[1],-w[2]/w[1],lty=2)
		abline((b-1)/w[1],-w[2]/w[1],lty=2)
	baseLearner <- c(baseLearner, paste0('ksvm', i))
	ime <- paste0('ksvm', i, '.rds')
	setwd(OkoljeBaseLearner)
	saveRDS(svp, ime) 
	}
	
	
	
	## Output profili
	OP_meta <- data.frame(matrix(NA, length(yMetaSet), 0))
	OP_test <- data.frame(matrix(NA, length(yTestSet), 0))
	
	for(bl in baseLearner){
		imeBL <-  paste0(bl, '.rds')
		imeStolpca <- paste0(bl, '_', levels(factor(yTrainBL)))
		setwd(OkoljeBaseLearner)
		model <- readRDS(imeBL)
		napoved <- predict(model, metaSet, type = 'prob')
		OP_meta[, imeStolpca] <- napoved
		napoved <- predict(model, testSet, type = 'prob')
		OP_test[, imeStolpca] <- napoved
	}
	

	setwd(OkoljeRezultatOutputProfile)
	saveRDS(OP_meta, 'OP_meta_verjetnost.rds')
	saveRDS(OP_test, 'OP_test_verjetnost.rds')
	
	
	
	## izbror ensembla KNORA 
	kNN_RC <- get.knnx(OP_meta, OP_test,  k=Kp, algorith = 'kd_tree')
	kNN_RC <- get.knnx(metaSet, testSet,  k=Kp, algorith = 'kd_tree')
						## algorithm=c("kd_tree", "cover_tree", "brute")
	matrika_sosediOP <- kNN_RC$nn.index
	
	
	predClass_meta <- probToClass(OP_meta, baseLearner, classes = levels(yMetaSet))
	predClass_test <- probToClass(OP_test, baseLearner, classes = levels(yTestSet))
	
	napoved <- NULL
	for(i in 1:nrow(predClass_test)){
		# print(i)
		najboljsi <- NULL
		meja <- 0
		for(mdl in 1:ncol(predClass_meta)){
			pravilnost <- sum(predClass_meta[matrika_sosediOP[i,],mdl] == yMetaSet[matrika_sosediOP[i,]])/Kp
			# pravilnost <- (predClass_meta[matrika_sosediOP[i,],mdl] == yTestSet[matrika_sosediOP[i,]])
			# print(pravilnost)
			if(meja < pravilnost){
				najboljsi <- mdl
				meja <- pravilnost
			}else if(meja == pravilnost){
				najboljsi <- c(najboljsi, mdl)
			}
		}
		napoved <- c(napoved,names(sort(table(as.numeric(predClass_test[i,najboljsi])), decreasing = TRUE)[1]))
	}
	napoved <- factor(napoved)
	
	confusionMatrix(table(napoved, yTestSet))
	
	predClass[i,]
	
	sosedi_classOP <-  
	
	f4 <- as.numeric(sosedi_classOP == ySosedSet[sosediOP])
	
	
	
	
	
	
	
	
	
	
	
	
	