## funkcija ki da v primerno obliko napovedi za 


setwd('C:/Users/Podlogar/Documents/DynamicEnsembleSelection/Podatki/Podatki')

podatki <- readRDS('razdelitevPodatkovIzbrani_list.rds')
izbrani <- colnames(podatki$xTest)

podatki <- read.csv('modelLibrary.csv')


setwd('C:/Users/Podlogar/Documents/DynamicEnsembleSelection/Podatki/Podatki_loceni')
for(imeS in izbrani){
	print(imeS)
	imeRDS <- paste0(imeS, '.rds')
	saveRDS(podatki[, imeS], imeRDS)
}


setwd('C:/Users/Podlogar/Documents/DynamicEnsembleSelection/Podatki/Podatki_loceni')


imenaStolpcev <- paste0(sort(rep(izbrani,2)), '_', c('neg', 'pos'))
matrikaNapovedi <- data.frame(matrix(NA, 1063347, length(imenaStolpcev)))
names(matrikaNapovedi) <- imenaStolpcev
for(imeS in izbrani){
	print(imeS)
	ime <- paste0(imeS, '_', c('neg', 'pos'))
	imeRDS <- paste0(imeS, '.rds')
	verjetnostPos <- readRDS(imeRDS)
	matrikaNapovedi[,ime] <- cbind(1-verjetnostPos, verjetnostPos)
}


library(pryr)
object_size(matrikaNapovedi)



imenaStolpcev <- paste0(sort(rep(izbrani,2)), '_', c('neg', 'pos'))
matrikaNapovedi <- NULL
for(imeS in izbrani){
	print(imeS)
	ime <- paste0(imeS, '_', c('neg', 'pos'))
	imeRDS <- paste0(imeS, '.rds')
	verjetnostPos <- readRDS(imeRDS)
	matrikaNapovedi <- cbind(matrikaNapovedi, cbind(1-verjetnostPos, verjetnostPos))
	#colnames(matrikaNapovedi) <- c(colnames(matrikaNapovedi), ime)
}
