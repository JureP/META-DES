## cross validation iskanje opt parametrvo
OkoljeIzvedba <- 'C:/Users/Podlogar/Documents/DES/Pima'
OkoljeFunkcije <- paste0(OkoljeIzvedba,'/Funkcije')
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



## base learnerji
baseLearner <- c('plr', 'nb', 'LogitBoost', 'rf', 'nnet', 'knn', 'gbm', 'lda', 'C5.0', 
		'multinom', 'pls', 'pcaNNet')
		

## podatki
data(PimaIndiansDiabetes)
osnovniPodatki <- na.omit(PimaIndiansDiabetes)
summary(osnovniPodatki)
onsovniPodatki_response <- osnovniPodatki[, 9]
onsovniPodatki_FM <- osnovniPodatki[, -9]

## razdelitev podatkov 

cvRazdelitev <- createFolds(onsovniPodatki_response, k = 4, list = TRUE, returnTrain = FALSE)
# vect <- 1:length(names(cvRazdelitev))
# for(i in 1:length(vect)){
	# vect <- c(vect[length(vect)], vect[-length(vect)])
	# print(vect)
# }
vect <- 1:length(names(cvRazdelitev))
for(i in 1:length(vect)){
	vect <- c(vect[length(vect)], vect[-length(vect)])
	razvrstitev <- names(cvRazdelitev[vect])
	## 
	trainBL <- onsovniPodatki_FM[cvRazdelitev[[razvrstitev[1]]], ]
	yTrainBL <- onsovniPodatki_response[cvRazdelitev[[razvrstitev[1]]]]
	
	sosedSet <- onsovniPodatki_FM[cvRazdelitev[[razvrstitev[2]]], ]
	ySosedSet <- onsovniPodatki_response[cvRazdelitev[[razvrstitev[2]]]]
	
	metaSet <- onsovniPodatki_FM[cvRazdelitev[[razvrstitev[3]]], ]
	yMetaSet <- onsovniPodatki_response[cvRazdelitev[[razvrstitev[3]]]]
	
	validSet <- onsovniPodatki_FM[cvRazdelitev[[razvrstitev[4]]], ]
	yValidSet <- onsovniPodatki_response[cvRazdelitev[[razvrstitev[4]]]]
	
		
	## mape v kater se shranjujejo rezultati
	OkoljeRezultat <- paste0(OkoljeIzvedba, '/Rezultati')
	OkoljeRezultatCV <- paste0(OkoljeRezultat, '/CV', i)
	######################################################################
	OkoljePodatki <- paste0(OkoljeRezultatCV, '/Osnovni podatki')
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
	dir.create(OkoljePodatki)
	dir.create(OkoljeBaseLearner)
	dir.create(OkoljeRezultatOutputProfile)
	dir.create(OkoljeMetaPrblm)
	dir.create(OkoljeMetaKlasifikator)
	dir.create(OkoljeMetaPrblm_VALID)
	dir.create(OkoljeKompetentnost)
	dir.create(OkoljeNapovedEnsemble)
	dir.create(OkoljeAccuracy)
	dir.create(OkoljeAccuracyBL)
	
	
	
	setwd(OkoljePodatki)
	saveRDS(trainBL, 'trainBL.rds')
	saveRDS(yTrainBL, 'yTrainBL.rds')
	
	saveRDS(sosedSet, 'sosedSet.rds')
	saveRDS(ySosedSet, 'ySosedSet.rds')
	
	saveRDS(metaSet, 'metaSet.rds')
	saveRDS(yMetaSet, 'yMetaSet.rds')
	
	saveRDS(validSet, 'validSet.rds')
	saveRDS(yValidSet, 'yValidSet.rds')

	# output profile na sosedSet
	# setwd(OkoljeRezultatOutputProfile)
	# OP_sosedi <- readRDS('OP_sosedi_verjetnost.rds')
	# output profile na metaSet
	# OP_meta <- readRDS('OP_meta_verjetnost.rds')
	# output profile na validSet
	# OP_valid <- readRDS('OP_meta_verjetnost.rds')



	## Delovanje probToClass
	# OP_metaNapoved <- readRDS('OP_meta_napoved.rds')
	# matrikaNapovedi <- probToClass(OP_meta, baseLearner, classes = levels(ySosedSet))
	# all(OP_metaNapoved == matrikaNapovedi)
		
		## region of competence izracunaj samo enkrat izberi relavantne.


				optMETA_DES_CARET(## parametri potrebni za ucenje base learnerja
							baseLearner, ## ime base learnerjev, ki jih podpira caret
							trainBL, ## podatki za ucenje base learnerja
							yTrainBL, ## respns vektor za ucenje base learnerja
							OkoljeBaseLearner, ## okolje kamor se shranijo base learner modeli
							OkoljeRezultatOutputProfile, ## okolje kamor se shranijo output profili

							## parametri sestavljanja meta problema ###########
							imenaMnozic = c(razvrstitev[1],razvrstitev[2], razvrstitev[3], razvrstitev[4]),  ## ime mnozice: [sosede katerih iscemo, iz katere so sosedje]
							metaSet, ## mnozica iz katere se sestavi meta problem
							yMetaSet, ## response vektor meta seta
								### 'n' za sestavitev metaFM, brez meta respons vektorja 
							sosedSet, ## mnozica iz katere so izbrane sosedi za metaSet
							ySosedSet, ## response vektor sosed seta
							# OP_meta, ## output profile (verjetnosti) za mnozico metaSet 
								## imena stolpcev: baseLearer1_class(1) baseLearner1_ class(2) ... baseLearner1_class(n) ... baseLearnerK_class(n)
							# OP_sosedi, ## output profile (verjetnosti) za mnozico sosedSet
								## imena stolpcev: baseLearer1_class(1) baseLearner1_ class(2) ... baseLearner1_class(n) ... baseLearnerK_class(n)
							K = c(5,10), ## stevilo sosedov za region of competence (vektor)
							Kp = c(5,10), ## stevilo sosedov iz output profile (vektor)
							knnALG = "kd_tree", ## algoritem po katerem get.knnx isce sosede
							OkoljeMetaPrblm, ## okolje kamor naj se shranijo matrike meta problemov
							OkoljeSosedi = paste0(OkoljeMetaPrblm, '/matrikaSosedi'), ## okolje kamor naj se shrani matrike sosedov
							
							## dodatni parametri ucenja meta klasifikatorja
							hC = c(1), ## meja razlicnosti napovedi baseLearnerjev za vkljucitev v podatke za meta ucenje
							hC_ONE = c(0.8,1), ## hC za model z enim meta klasifikatorjem ta vsak bl
							MetaMode = c('individual', 'one'), ## 'one'
							metaALG = c('rf', 'gbm'), ## algoriti uproabljeni za ucenje meta problema
							OkoljeMetaKlasifikator, ## okolje kamor se shranijo meta klasifikatorji
							
							## dodatni parametri za napovedovanje META-DES
							imenaMnozicVALID = c(razvrstitev[1], razvrstitev[3], razvrstitev[4]),
							validSet,
							yValidSet,
							# OP_valid,
							sosedSet_VALID = metaSet,
							ySosedSet_VALID = yMetaSet,
							# OP_sosedi_VALID = OP_meta,
							OkoljeMetaPrblm_VALID,
							
							## napovedi meta klasifikatorja in ensembla
							OkoljeKompetentnost, ## okolje kamor se shranijo matrike kompetentnosti
							kompetThrsh = c(0.5, 0.8), ## meja kompetentnosti za vklucitev v ensemble
							OkoljeNapovedEnsemble ## okolje kamor se shranijo napovedi ensembla
							)
							
	## ucenje base learnerja na celotnih podatkih
	for(bl in baseLearner){
		setwd(OkoljeBaseLearner)
		ime <- paste0(bl, 'VSI_PODATKI.rds')
		if(ime %in% dir()){
			print(paste(bl, 'ze naucen'))
		}else{
			FM <- rbind(trainBL, sosedSet, metaSet)
			y <- factor(c(as.character(yTrainBL), as.character(ySosedSet), as.character(yMetaSet)))
			BLmodel <- train(FM, y, method = bl)
			setwd(OkoljeBaseLearner)
			ime <- paste0(bl, 'VSI_PODATKI.rds')
			saveRDS(BLmodel, ime)
		}
	}
	
	

		
	
}


	### NAPOVEDI ORACLA!!!
		
	# setwd(OkoljeRezultatOutputProfile)
	# OP_valid <- readRDS('OP_valid_verjetnost.rds')
	# for(bl in baseLearner){
		# ime <- paste0(bl, '_pos')
		# napoved <- OP_valid[,ime]
		# napovedClass <- napoved 
		# napovedClass[napoved < 0.5] <- 'neg'
		# napovedClass[napoved >= 0.5] <- 'pos'
		# napovedClass <- factor(napovedClass)
		# setwd(OkoljeAccuracyBL)
		# saveRDS(confusionMatrix(table(napovedClass, yValidSet)), paste0(bl. '_acc.rds'))
	# }
	

	



## uspesnost napovedi na cv

{## individual
	povprecjeAccuracy <- data.frame(matrix(NA, , 9))
	colnames(povprecjeAccuracy) <- c('K', 'Kp', 'knnALG', 'metaALG', 'hC', 'kompetThrsh', 'Accuracy', 'Sensitivity', 'Specificity')


	matrikaAccuracy <- NULL
	matrikaSensitivity <- NULL
	matrikaSpecificity <- NULL
	matrike <- list()
	vect <- 1:length(names(cvRazdelitev))
	for(i in 1:length(vect)){
		OkoljeRezultat <- paste0(OkoljeIzvedba, '/Rezultati')
		OkoljeRezultatCV <- paste0(OkoljeRezultat, '/CV', i)
		OkoljeAccuracy <- paste0(OkoljeRezultatCV, '/Accuracy')
		setwd(OkoljeAccuracy)
		matrika <- readRDS(dir()[substr(dir(), 1, 17) == 'accuracy[trainBL]'])
		matrike[[i]] <- matrika
		matrikaAccuracy <- cbind(matrikaAccuracy, as.numeric(matrika$Accuracy))
		matrikaSensitivity <- cbind(matrikaSensitivity, as.numeric(matrika$Sensitivity))
		matrikaSpecificity <- cbind(matrikaSpecificity, as.numeric(matrika$Specificity))
	}
	povprecjeAccuracy <- cbind(matrika[,c('K', 'Kp', 'knnALG', 'metaALG', 'hC', 'kompetThrsh')],
								rowMeans(matrikaAccuracy), rowMeans(matrikaSensitivity), rowMeans(matrikaSpecificity))
	colnames(povprecjeAccuracy) <- c('K', 'Kp', 'knnALG', 'metaALG', 'hC', 'kompetThrsh', 'Accuracy', 'Sensitivity', 'Specificity')
}

{## one
	povprecjeAccuracyONE <- data.frame(matrix(NA, , 9))
	colnames(povprecjeAccuracyONE) <- c('K', 'Kp', 'knnALG', 'metaALG', 'hC', 'kompetThrsh', 'Accuracy', 'Sensitivity', 'Specificity')


	matrikaAccuracy <- NULL
	matrikaSensitivity <- NULL
	matrikaSpecificity <- NULL
	matrike <- list()
	vect <- 1:length(names(cvRazdelitev))
	for(i in 1:length(vect)){
		OkoljeRezultat <- paste0(OkoljeIzvedba, '/Rezultati')
		OkoljeRezultatCV <- paste0(OkoljeRezultat, '/CV', i)
		OkoljeAccuracy <- paste0(OkoljeRezultatCV, '/Accuracy')
		setwd(OkoljeAccuracy)
		matrika <- readRDS(dir()[substr(dir(), 1, 11) == 'accuracyONE'])
		matrike[[i]] <- matrika
		matrikaAccuracy <- cbind(matrikaAccuracy, as.numeric(matrika$Accuracy))
		matrikaSensitivity <- cbind(matrikaSensitivity, as.numeric(matrika$Sensitivity))
		matrikaSpecificity <- cbind(matrikaSpecificity, as.numeric(matrika$Specificity))
	}
	povprecjeAccuracyONE <- cbind(matrika[,c('K', 'Kp', 'knnALG', 'metaALG', 'hC', 'kompetThrsh')],
								rowMeans(matrikaAccuracy), rowMeans(matrikaSensitivity), rowMeans(matrikaSpecificity))
	colnames(povprecjeAccuracyONE) <- c('K', 'Kp', 'knnALG', 'metaALG', 'hC', 'kompetThrsh', 'Accuracy', 'Sensitivity', 'Specificity')
}

{## napovedi base learnerjev uporabljenih v ensemblu
	matrikaAcc_BL <- data.frame(matrix(NA, length(baseLearner),4))
	rownames(matrikaAcc_BL) <- baseLearner
	colnames(matrikaAcc_BL) <- c('CV1','CV2','CV3','CV4')
	vect <- 1:length(names(cvRazdelitev))
	for(i in 1:length(vect)){
		OkoljeRezultat <- paste0(OkoljeIzvedba, '/Rezultati')
		OkoljeRezultatCV <- paste0(OkoljeRezultat, '/CV', i)
		OkoljePodatki <- paste0(OkoljeRezultatCV, '/Osnovni podatki')
		OkoljeRezultatOutputProfile <- paste0(OkoljeRezultatCV, '/Napovedi BL')
		setwd(OkoljePodatki)
		yValid <- readRDS('yValidSet.rds')
		setwd(OkoljeRezultatOutputProfile)
		OP_valid <- readRDS('OP_valid_verjetnost.rds')
		napovedClass<- probToClass(OP_valid, baseLearner, levels(yValid))
		for(bl in baseLearner){
			matrikaAcc_BL[bl, i] <- confusionMatrix(table(napovedClass[,bl],yValid))[[3]]['Accuracy']
		}
	}
}


{## napovedi base learnejev naucenih na vseh podatkih ki so na voljo
	matrikaAcc_BL_VSI <- data.frame(matrix(NA, length(baseLearner),4))
	rownames(matrikaAcc_BL_VSI) <- baseLearner
	colnames(matrikaAcc_BL_VSI) <- c('CV1','CV2','CV3','CV4')
	vect <- 1:length(names(cvRazdelitev))
	for(i in 1:length(vect)){
		OkoljeRezultat <- paste0(OkoljeIzvedba, '/Rezultati')
		OkoljeRezultatCV <- paste0(OkoljeRezultat, '/CV', i)
		OkoljePodatki <- paste0(OkoljeRezultatCV, '/Osnovni podatki')
		OkoljeBaseLearner <- paste0(OkoljeRezultatCV, '/Base Learner')
		setwd(OkoljePodatki)
		yValid <- readRDS('yValidSet.rds')
		validSet <- readRDS('validSet.rds')
		setwd(OkoljeBaseLearner)
		for(bl in baseLearner){
			imeBL <- paste0(bl,'VSI_PODATKI.rds')
			model <- readRDS(imeBL)
			OP_valid <- predict(model, validSet, type = 'prob')
			colnames(OP_valid) <- paste0(bl, '_', levels(yValid))
			napovedClass <- probToClass(OP_valid, bl, levels(yValid))
			matrikaAcc_BL_VSI[bl, i] <- confusionMatrix(table(napovedClass[,bl],yValid))[[3]]['Accuracy']
		}
	}
}


{ ## oracle 
	for(i in 1:length(vect)){
		OkoljeRezultat <- paste0(OkoljeIzvedba, '/Rezultati')
		OkoljeRezultatCV <- paste0(OkoljeRezultat, '/CV', i)
		OkoljePodatki <- paste0(OkoljeRezultatCV, '/Osnovni podatki')
		OkoljeRezultatOutputProfile <- paste0(OkoljeRezultatCV, '/Napovedi BL')
		setwd(OkoljePodatki)
		yValid <- readRDS('yValidSet.rds')
		setwd(OkoljeRezultatOutputProfile)
		OP_valid <- readRDS('OP_valid_verjetnost.rds')
		napovedClass<- probToClass(OP_valid, baseLearner, levels(yValid))
		oracle <- apply(napovedClass, 2, function(x){x == yValid})
		ORACLE_LIMIT <- sum(apply(oracle, 1, any))/nrow(oracle)
		print(ORACLE_LIMIT)
}


povprecjeAccuracy
povprecjeAccuracyONE
rowMeans(matrikaAcc)
rowMeans(matrikaAcc_BL_VSI)



max(rowMeans(matrikaAcc))
max(rowMeans(matrikaAcc_BL_VSI))
max(povprecjeAccuracy$Accuracy)
max(povprecjeAccuracyONE$Accuracy)

mean(povprecjeAccuracy$Accuracy)
mean(povprecjeAccuracyONE$Accuracy)


## napovedi oracla
## napoved bl-ja naucenega na celotnih podatkih
## set seed




