## META-DES financni podatki
set.seed(10)

OkoljePodatki <- 'C:/Users/Podlogar/Documents/DES/Financni/Podatki'

OkoljeIzvedba <- 'C:/Users/Podlogar/Documents/DES/Financni'
OkoljeFunkcije <- paste0(OkoljeIzvedba,'/Funkcije')
## imena mnozic za potrebe poimenovanja.
imenaMnozic <- c('trainBL', 'sosedSet', 'metaSet', 'testSet')
imenaMnozicTest <- imenaMnozic[c(1,3,4)]

{## nastavitve parametrov
MetaMode = c('individual', 'one')
K = 100 ## stevilo sosedov za region of competence (vektor)
Kp = 100 ## stevilo sosedov iz output profile (vektor)
knnALG = "kd_tree" ## algoritem po katerem get.knnx isce sosede
metaALG = c('rf', 'elm') ## klasifikacijski algoritmi uporabljeni pri ucenju meta klasifikatorja
hC = c(0.7, 1)
hC_ONE = c(0.7, 1)
kompetThrsh = c(0.7, 0.9)
}

baseLearner <- c('plr', 'nb', 'LogitBoost', 'nnet', 'knn', 'gbm', 'lda', 
				'multinom', 'pls', 'pcaNNet', 'rf')


baseLearner <- c('plr', 'nb', 'LogitBoost', 'rf', 'nnet', 'knn', 'gbm', 'lda', 'C5.0', 
				'multinom', 'pls', 'pcaNNet')
				
	## python			
	# metaClassifiers = [lr, elm]
    # hCs = [1.0, 0.5]
    # nrNeigh = [305]#, 1000, 3000]
    # modes = ["weighted"]
    # metrics = ["l2", "chebyshev"]#BallTree.valid_metrics
    # metaClsModes = ["combined"]
    # normalizeMetaFeatures = [True, False]
    # competenceTressholds = [0.4,0.5,0.6]
	
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
	trainBL <- readRDS('trainBL.rds')
	yTrainBL <- readRDS('yTrainBL.rds')
	
	sosedSet <- readRDS('sosedSet.rds')
	ySosedSet <- readRDS('ySosedSet.rds')
	
	metaSet <- readRDS('metaSet.rds')
	yMetaSet <- readRDS('yMetaSet.rds')
	
	testSet <- readRDS('testSet.rds')
	yTestSet <- readRDS('yTestSet.rds')
}
		
{## mape v kater se shranjujejo rezultati
	OkoljeRezultat <- paste0(OkoljeIzvedba, '/Rezultati_nelinearni')
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
}	
	
	## ucenje pool of classifiers
	for(bl in baseLearner){
		setwd(OkoljeBaseLearner)
		ime <- paste0(bl, '.rds')
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


	## NAPOVEDI BASE LEARNERJEV ###############################################
	###########################################################################						
	OP_sosedi <- data.frame(matrix(NA, length(ySosedSet), 0))
	OP_meta <- data.frame(matrix(NA, length(yMetaSet), 0))
	OP_test <- data.frame(matrix(NA, length(yTestSet), 0))
	
	for(bl in baseLearner){
		imeBL <-  paste0(bl, '.rds')
		imeStolpca <- paste0(bl, '_', levels(factor(yTrainBL)))
		setwd(OkoljeBaseLearner)
		model <- readRDS(imeBL)
		napoved <- predict(model, newdata = sosedSet, type = 'prob')
		OP_sosedi[, imeStolpca] <- napoved
		napoved <- predict(model, metaSet, type = 'prob')
		OP_meta[, imeStolpca] <- napoved
		napoved <- predict(model, testSet, type = 'prob')
		OP_test[, imeStolpca] <- napoved
	}
	

	setwd(OkoljeRezultatOutputProfile)
	saveRDS(OP_sosedi, 'OP_sosedi_verjetnost.rds')
	saveRDS(OP_meta, 'OP_meta_verjetnost.rds')
	saveRDS(OP_test, 'OP_test_verjetnost.rds')
	
	
	
	## SESTAVLJANJE META PROBLEMA #############################################
	###########################################################################
	## in: SosedSed, MetaSet, OP_SosedSet?, OP_MetaSet,
	##		K, Kp, metrika, folder, baseLearner
	## out: metaFM, metaResponse 
	
	## delovanje: pogleda v folder, ce so potrebne metaProblemi ze izracunani, 
	## ce da vrne ustrezno, sicer izracuna in shrani v folder
	
	## razlicni K, Kp, hC, nacini iskanja sosedov
	## za vsak K, Kp svoj problem, 
	## hC pobres ven iz meta problema relavantne
	metaProblemSAVE(imenaMnozic[1:3], ## ime mnozice: [sosede katerih iscemo, iz katere so sosedje]
						metaSet, ## mnozica iz katere se sestavi meta problem
						yMetaSet, ## response vektor meta seta
							### 'n' za sestavitev metaFM, brez meta respons vektorja 
						sosedSet, ## mnozica iz katere so izbrane sosedi za metaSet
						ySosedSet, ## response vektor sosed seta
						baseLearner, ## ime base learnerjev (vektor)
						OP_meta, ## output profile (verjetnosti) za mnozico metaSet 
							## imena stolpcev: baseLearer1_class(1) baseLearner1_ class(2) ... baseLearner1_class(n) ... baseLearnerK_class(n)
						OP_sosedi, ## output profile (verjetnosti) za mnozico sosedSet
							## imena stolpcev: baseLearer1_class(1) baseLearner1_ class(2) ... baseLearner1_class(n) ... baseLearnerK_class(n)
						K, ## stevilo sosedov za region of competence (vektor)
						Kp, ## stevilo sosedov iz output profile (vektor)
						knnALG, ## algoritem po katerem get.knnx isce sosede
						OkoljeMetaPrblm, ## okolje kamor naj se shranijo matrike meta problemov
						paste0(OkoljeMetaPrblm, '/matrikaSosedi') ## okolje kamor naj se shrani matrike sosedov
						)
	

	## UCENJE META KLASIFIKATORJA #############################################
	###########################################################################
	## za razlicne K,Kp, knnALG
	## za razlicne hC, razlicni klasifikatorji [log reg, elm, ...], mode = one ali individual
	
	## imena mnozic 
	## stevec poteka: izracunal 50% primerov
	############################
	
	
	## model za vsak baseLearner
	if('individual' %in% MetaMode){
		############################
		ucenjeMetaKlasifikator(imenaMnozic[1:3], ## ime mnozice: [sosede katerih iscemo, iz katere so sosedje]
							OP_meta, ## output profile (verjetnosti) za mnozico metaSet 
									## imena stolpcev: baseLearer1_class(1) baseLearner1_ class(2) ... baseLearner1_class(n) ... baseLearnerK_class(n)
							classOsnovni =  levels(ySosedSet), ## imena classov pri osnovnem problemu kot vektor (npr: c('neg', 'pos')
							baseLearner, ## ime base learnerjev (vektor)
							K, ## stevilo sosedov za region of competence (vektor)
							Kp, ## stevilo sosedov iz output profile (vektor)
							knnALG, ## algoritem po katerem get.knnx isce sosede
							metaALG, ## klasifikacijski algoritmi uporabljeni pri ucenju meta klasifikatorja
							hC, ## kako razlicne morajo biti napovedi base learnerjev, za x iz metaSet da ga vkljucimo v ucenje
							OkoljeMetaPrblm, ## okolje kjer so shranjene metaProblemi 
							OkoljeMetaKlasifikator ## okolje, kamor se shranijo meta klasifikatorji
							)
		############################		
	}

	if('one' %in% MetaMode){
	############################
	ucenjeMetaKlasifikatorONE(imenaMnozic[1:3], ## ime mnozice: [sosede katerih iscemo, iz katere so sosedje]
						OP_meta, ## output profile (verjetnosti) za mnozico metaSet 
								## imena stolpcev: baseLearer1_class(1) baseLearner1_ class(2) ... baseLearner1_class(n) ... baseLearnerK_class(n)
						classOsnovni =  levels(ySosedSet), ## imena classov pri osnovnem problemu kot vektor (npr: c('neg', 'pos')
						baseLearner, ## ime base learnerjev (vektor)
						K, ## stevilo sosedov za region of competence (vektor)
						Kp, ## stevilo sosedov iz output profile (vektor)
						knnALG, ## algoritem po katerem get.knnx isce sosede
						metaALG, ## klasifikacijski algoritmi uporabljeni pri ucenju meta klasifikatorja
						hC = hC_ONE, ## kako razlicne morajo biti napovedi base learnerjev, za x iz metaSet da ga vkljucimo v ucenje
						OkoljeMetaPrblm, ## okolje kjer so shranjene metaProblemi 
						OkoljeMetaKlasifikator ## okolje, kamor se shranijo meta klasifikatorji
							)
	############################
	}
	
	## IZRACUN MATRIKE KOMPETENTNOSTI #########################################
	###########################################################################
	
	## izracunaj metaFM na testSet, shrani v mapo metaFM testSet
	
	## preglej retultate vseh meta klasifikatorjev kombinacij (za bl) 
	
	
	## ALI JE SMISELNO SHRANJEVATI metaProbleme (cas loadanja vs cas racunanja) 30/1 sekund
	## IZRACUN IN SHRANITEV metaFM
	
	
	metaProblemSAVE(imenaMnozicTest, ## ime mnozice: [sosede katerih iscemo, iz katere so sosedje]
					metaSet = testSet, ## mnozica iz katere se sestavi meta problem
					yMetaSet = yTestSet, ## response vektor meta seta
						### 'n' za sestavitev metaFM, brez meta respons vektorja 
					sosedSet = metaSet, ## mnozica iz katere so izbrane sosedi za metaSet
					ySosedSet = yMetaSet, ## response vektor sosed seta
					baseLearner, ## ime base learnerjev (vektor)
					OP_meta = OP_test, ## output profile (verjetnosti) za mnozico metaSet 
						## imena stolpcev: baseLearer1_class(1) baseLearner1_ class(2) ... baseLearner1_class(n) ... baseLearnerK_class(n)
					OP_sosedi = OP_meta, ## output profile (verjetnosti) za mnozico sosedSet
						## imena stolpcev: baseLearer1_class(1) baseLearner1_ class(2) ... baseLearner1_class(n) ... baseLearnerK_class(n)
					K, ## stevilo sosedov za region of competence (vektor)
					Kp, ## stevilo sosedov iz output profile (vektor)
					knnALG, ## algoritem po katerem get.knnx isce sosede
					OkoljeMetaPrblm = OkoljeMetaPrblm_VALID, ## okolje kamor naj se shranijo matrike meta problemov
					OkoljeSosedi = paste0(OkoljeMetaPrblm_VALID, '/matrikaSosedi') ## okolje kamor naj se shrani matrike sosedov
					)
					

	if('individual' %in% MetaMode){
		## izracun kompetentnosti klasifikatorja (vsak)
		##############################################
		for(nSosedi in K){
			for(OPnSosedi in Kp){
				for(alg in knnALG){
					for(algM in metaALG){
						for(meja in hC){
							kompetentnost <- NULL
							for(bl in baseLearner){
								## loadanje metaProblema
								setwd(OkoljeMetaPrblm_VALID)
								imeMetaProblem <- paste0('matrikaProblem[BL]', bl,'[trainBL]', imenaMnozicTest[1],
														'[sosedSet]',imenaMnozicTest[2], '[metaSet]',imenaMnozicTest[3],
														'[K]',nSosedi, '[Kp]', OPnSosedi, '[knnALG]', alg, '.rds')
								metaFM <- readRDS(imeMetaProblem)$metaFM
								## loadanje meta klasifikatorja
								setwd(OkoljeMetaKlasifikator)
								imeMetaKlasifikator <- paste0('metaKlasifikator[BL]', bl,'[trainBL]', imenaMnozic[1],'[sosedSet]',
																	imenaMnozic[2],'[metaSet]', imenaMnozic[3], '[K]',nSosedi, 
																	'[Kp]',OPnSosedi, '[knnALG]', alg, '[metaALG]', algM, '[cH]', meja, '.rds')
								metaKlasifikator <- readRDS(imeMetaKlasifikator)
								## napovedi meta klasifikatorja
								napoved <- predict(metaKlasifikator, metaFM, type = 'prob')[2]
								kompetentnost <- cbind(kompetentnost, as.matrix(napoved))
							}
							kompetentnost <- data.frame(kompetentnost)
							colnames(kompetentnost) <- baseLearner
							## shranjevanje matrike kompetentnosti
							setwd(OkoljeKompetentnost)
							imeKompetentnost <- paste0('kompetentnost[trainBL]', imenaMnozic[1],'[sosedSet]',
																	imenaMnozic[2], '[metaSet]',imenaMnozic[3],
																	'[testSet]',imenaMnozic[4], '[K]',nSosedi, 
																	'[Kp]',OPnSosedi, '[knnALG]', alg, 
																	'[metaALG]', algM, '[cH]', meja, '.rds')
							saveRDS(kompetentnost, imeKompetentnost)
						}				
					}
				}
			}
		}

## pravilnost napvoedi meta klasifikatorja		
# N <- kompetentnost[,1]
# N[kompetentnost[,1] < 0.7] <- 'N'
# N[kompetentnost[,1] >= 0.7] <- 'Y'
# N <- factor(N[,1])
# table(N, metaProblem$metaClass)
# confusionMatrix(table(N, metaProblem$metaClass))		

		## napovedi ensembla
		##############################################

		for(nSosedi in K){
			for(OPnSosedi in Kp){
				for(alg in knnALG){
					for(algM in metaALG){
						for(meja in hC){
							## loadanje matrik kompetentnosti
							setwd(OkoljeKompetentnost)
							imeKompetentnost <- paste0('kompetentnost[trainBL]', imenaMnozic[1],'[sosedSet]',
																	imenaMnozic[2], '[metaSet]',imenaMnozic[3],
																	'[testSet]',imenaMnozic[4], '[K]',nSosedi, 
																	'[Kp]',OPnSosedi, '[knnALG]', alg, 
																	'[metaALG]', algM, '[cH]', meja, '.rds')
							kompetentnost <- readRDS(imeKompetentnost)
							## sestavljanje napovedi ensembla
							imena <- paste0(baseLearner, '_', levels(yMetaSet)[1])
							napovedBL <- OP_test[, imena]
							for(k in kompetThrsh){
								ensembleNapoved <- napovedEnsemble(napovedBL,
																	kompetentnost,
																	k,
																	bup = 4)
								setwd(OkoljeNapovedEnsemble)
								imeNapovedEns <- paste0('napovedEns[trainBL]', imenaMnozic[1],'[sosedSet]',
																	imenaMnozic[2], '[metaSet]',imenaMnozic[3],
																	'[testSet]',imenaMnozic[4], '[K]',nSosedi, 
																	'[Kp]',OPnSosedi, '[knnALG]', alg, 
																	'[metaALG]', algM, '[cH]', meja, '.rds')
								saveRDS(ensembleNapoved,imeNapovedEns)
							}
						}				
					}
				}
			}
		}
		

		## pravilnost napovedi, shrani v eno matriko (vsi parametri)
		##############################################
		imeStolpcev <- c('trainBL' ,'sosedSet', 'metaSet', 'testSet', 'K', 'Kp', 'knnALG', 
					'metaALG', 'hC', 'kompetThrsh', 'Accuracy', 'Sensitivity', 'Specificity')
		rezultati <- data.frame(matrix(NA, 0,length(imeStolpcev)))
		colnames(rezultati) <- imeStolpcev
		
		for(nSosedi in K){
			for(OPnSosedi in Kp){
				for(alg in knnALG){
					for(algM in metaALG){
						for(meja in hC){
							for(k in kompetThrsh){
								setwd(OkoljeNapovedEnsemble)
								imeNapovedEns <- paste0('napovedEns[trainBL]', imenaMnozic[1],'[sosedSet]',
																	imenaMnozic[2], '[metaSet]',imenaMnozic[3],
																	'[testSet]',imenaMnozic[4], '[K]',nSosedi, 
																	'[Kp]',OPnSosedi, '[knnALG]', alg, 
																	'[metaALG]', algM, '[cH]', meja, '.rds')
								napovedEnsembla <- readRDS(imeNapovedEns)
								napovedClass <- napovedEnsembla
								thr <- 0.5
								napovedClass[napovedEnsembla < thr] <- levels(yMetaSet)[2]
								napovedClass[napovedEnsembla >= thr] <- levels(yMetaSet)[1]
								napovedClass <- factor(napovedClass)
								A <- confusionMatrix(table(napovedClass, yTestSet))
								# print(A)
						
								rezultati[nrow(rezultati)+1,] <- c(imenaMnozic[1], imenaMnozic[2], imenaMnozic[3],
																	imenaMnozic[4],	nSosedi, OPnSosedi, alg, algM,
																	meja, k, A[[3]]['Accuracy'], A[[4]]['Sensitivity'],
																	A[[4]]['Specificity'])	
							}
						}				
					}
				}
			}
		}
			

		
		setwd(OkoljeAccuracy)
		ime <- paste0('accuracy[trainBL]', imenaMnozic[1],'[sosedSet]',	imenaMnozic[2],
						'[metaSet]',imenaMnozic[3], '[testSet]',imenaMnozic[4],'.rds')
		saveRDS(rezultati, ime)
	}
	
	if('one' %in% MetaMode){
		## izracun kompetentnosti klasifikatorja (vsak)
		##############################################
		for(nSosedi in K){
			for(OPnSosedi in Kp){
				for(alg in knnALG){
					for(algM in metaALG){
						for(meja in hC_ONE){
							kompetentnost <- NULL
							for(bl in baseLearner){
								## loadanje metaProblema
								setwd(OkoljeMetaPrblm_VALID)
								imeMetaProblem <- paste0('matrikaProblem[BL]', bl,'[trainBL]', imenaMnozicTest[1],
														'[sosedSet]',imenaMnozicTest[2], '[metaSet]',imenaMnozicTest[3],
														'[K]',nSosedi, '[Kp]', OPnSosedi, '[knnALG]', alg, '.rds')
								metaFM <- readRDS(imeMetaProblem)$metaFM
								## loadanje meta klasifikatorja
								setwd(OkoljeMetaKlasifikator)
								imeMetaKlasifikator <- paste0('metaKlasifikator[ONE]','[trainBL]', imenaMnozic[1],'[sosedSet]',
																	imenaMnozic[2],'[metaSet]', imenaMnozic[3], '[K]',nSosedi, 
																	'[Kp]',OPnSosedi, '[knnALG]', alg, '[metaALG]', algM, '[cH]', meja, '.rds')
								metaKlasifikator <- readRDS(imeMetaKlasifikator)
								## napovedi meta klasifikatorja
								napoved <- predict(metaKlasifikator, metaFM, type = 'prob')[2]
								kompetentnost <- cbind(kompetentnost, as.matrix(napoved))
							}
							kompetentnost <- data.frame(kompetentnost)
							colnames(kompetentnost) <- baseLearner
							## shranjevanje matrike kompetentnosti
							setwd(OkoljeKompetentnost)
							imeKompetentnost <- paste0('kompetentnostONE[trainBL]', imenaMnozic[1],'[sosedSet]',
																	imenaMnozic[2], '[metaSet]',imenaMnozic[3],
																	'[testSet]',imenaMnozic[4], '[K]',nSosedi, 
																	'[Kp]',OPnSosedi, '[knnALG]', alg, 
																	'[metaALG]', algM, '[cH]', meja, '.rds')
							saveRDS(kompetentnost, imeKompetentnost)
						}				
					}
				}
			}
		}
		

		
		## napovedi ensembla
		##############################################

		for(nSosedi in K){
			for(OPnSosedi in Kp){
				for(alg in knnALG){
					for(algM in metaALG){
						for(meja in hC_ONE){
							## loadanje matrik kompetentnosti
							setwd(OkoljeKompetentnost)
							imeKompetentnost <- paste0('kompetentnostONE[trainBL]', imenaMnozic[1],'[sosedSet]',
																	imenaMnozic[2], '[metaSet]',imenaMnozic[3],
																	'[testSet]',imenaMnozic[4], '[K]',nSosedi, 
																	'[Kp]',OPnSosedi, '[knnALG]', alg, 
																	'[metaALG]', algM, '[cH]', meja, '.rds')
							kompetentnost <- readRDS(imeKompetentnost)
							## sestavljanje napovedi ensembla
							imena <- paste0(baseLearner, '_', levels(yMetaSet)[1])
							napovedBL <- OP_test[, imena]
							for(k in kompetThrsh){
								ensembleNapoved <- napovedEnsemble(napovedBL,
																	kompetentnost,
																	k,
																	bup = 4)
								setwd(OkoljeNapovedEnsemble)
								imeNapovedEns <- paste0('napovedEnsONE[trainBL]', imenaMnozic[1],'[sosedSet]',
																	imenaMnozic[2], '[metaSet]',imenaMnozic[3],
																	'[testSet]',imenaMnozic[4], '[K]',nSosedi, 
																	'[Kp]',OPnSosedi, '[knnALG]', alg, 
																	'[metaALG]', algM, '[cH]', meja, '.rds')
								saveRDS(ensembleNapoved,imeNapovedEns)
							}
						}				
					}
				}
			}
		}
		

		## pravilnost napovedi, shrani v eno matriko (vsi parametri)
		##############################################
		imeStolpcev <- c('trainBL' ,'sosedSet', 'metaSet', 'testSet', 'K', 'Kp', 'knnALG', 
					'metaALG', 'hC', 'kompetThrsh', 'Accuracy', 'Sensitivity', 'Specificity')
		rezultati_ONE <- data.frame(matrix(NA, 0,length(imeStolpcev)))
		colnames(rezultati_ONE) <- imeStolpcev
		
		for(nSosedi in K){
			for(OPnSosedi in Kp){
				for(alg in knnALG){
					for(algM in metaALG){
						for(meja in hC_ONE){
							for(k in kompetThrsh){
								setwd(OkoljeNapovedEnsemble)
								imeNapovedEns <- paste0('napovedEnsONE[trainBL]', imenaMnozic[1],'[sosedSet]',
																	imenaMnozic[2], '[metaSet]',imenaMnozic[3],
																	'[testSet]',imenaMnozic[4], '[K]',nSosedi, 
																	'[Kp]',OPnSosedi, '[knnALG]', alg, 
																	'[metaALG]', algM, '[cH]', meja, '.rds')
								napovedEnsembla <- readRDS(imeNapovedEns)
								napovedClass <- napovedEnsembla
								thr <- 0.5
								napovedClass[napovedEnsembla < thr] <- levels(yMetaSet)[2]
								napovedClass[napovedEnsembla >= thr] <- levels(yMetaSet)[1]
								napovedClass <- factor(napovedClass)
								A <- confusionMatrix(table(napovedClass, yTestSet))
								# print(A)
						
								rezultati_ONE[nrow(rezultati_ONE)+1,] <- c(imenaMnozic[1], imenaMnozic[2], imenaMnozic[3],
																	imenaMnozic[4],	nSosedi, OPnSosedi, alg, algM,
																	meja, k, A[[3]]['Accuracy'], A[[4]]['Sensitivity'],
																	A[[4]]['Specificity'])	
							}
						}				
					}
				}
			}
		}
			

		
		setwd(OkoljeAccuracy)
		ime <- paste0('accuracyONE[trainBL]', imenaMnozic[1],'[sosedSet]',	imenaMnozic[2],
						'[metaSet]',imenaMnozic[3], '[testSet]',imenaMnozic[4],'.rds')
		saveRDS(rezultati_ONE, ime)
	}


## pravilnost napovedi oracla.
setwd(OkoljePodatki)
yTest <- readRDS('yTestSet.rds')
setwd(OkoljeRezultatOutputProfile)
OP_test <- readRDS('OP_test_verjetnost.rds')
napovedClass<- probToClass(OP_test, baseLearner, levels(yTest))
oracle <- apply(napovedClass, 2, function(x){x == yTest})
ORACLE_LIMIT <- sum(apply(oracle, 1, any))/nrow(oracle)
print(ORACLE_LIMIT)
	





yTestSet <- as.character(yTestSet)
yTestSet[yTestSet == 1] <- 'I'
yTestSet[yTestSet == 2] <- 'II'
## ifelse(as.character(yTestSet) == 1, 'I', 'II')
yTestSet <- factor(yTestSet)
## pravilnost napovedi base learnerjev
blAcc <- NULL
for(bl in baseLearner){
	print(bl)
	imeBL <-  paste0(bl, '.rds')
	setwd(OkoljeBaseLearner)
	model <- readRDS(imeBL)
	napoved <- predict(model, testSet)
	blAcc <- c(blAcc, confusionMatrix(table(napoved, yTestSet))[[3]]['Accuracy'])
	# print(confusionMatrix(table(napoved, yTestSet)))
}
names(blAcc) <- baseLearner
print(sort(blAcc))
	
rezultati
rezultati_ONE
ORACLE_LIMIT
	
	