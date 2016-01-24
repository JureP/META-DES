## P2 problem
set.seed(101)
library(caret)
library(kernlab)
OkoljePodatki <- 'C:/Users/Podlogar/Documents/DES/P2/Podatki'
OkoljeBL <- 'C:/Users/Podlogar/Documents/DES/P2/Base learner'
n <- 2*500 + 2*500 + 2*500 + 2*2000
FM <- matrix(runif(n), n/2, 2)



vect <- c(1, 0.4)



classP2 <- function(vect){
	if((vect[2] > (-0.1*(vect[1]*10)^2 + 0.6*sin(4*vect[1]*10) + 8)/10 & vect[2] > ((10*vect[1] -2)^2 + 1)/10) |
		(vect[2] < (sin(10*vect[1]) + 5)/10 & vect[2] > ((10*vect[1] -2)^2 + 1)/10) |
		(vect[2] > (sin(10*vect[1]) + 5)/10 & vect[2] < ((10*vect[1] -2)^2 + 1)/10 & vect[2] < (-0.1*(vect[1]*10)^2 + 0.6*sin(4*vect[1]*10) + 8)/10) |
		(vect[2] < (sin(10*vect[1]) + 5)/10 & vect[2] > (-0.1*(vect[1]*10)^2 + 0.6*sin(4*vect[1]*10) + 8)/10) |
		(vect[2] > (0.5*(10*vect[1] - 10)^2 + 6)/10)
		){
		return(1)
	}
	else{
		return(2)
	}
}


class <- apply(FM, 1, classP2)
summary(factor(class))


trainBL <- FM[1:500,]
yTrainBL <- factor(class[1:500])

sosedSet <- FM[501:1000,]
ySosedSet <- factor(class[501:1000])

metaSet <- FM[1001:1500,]
yMetaSet <- factor(class[1001:1500])

testSet <- FM[1501:3500,]
yTestSet <- factor(class[1501:3500])

setwd(OkoljePodatki)
	saveRDS(trainBL, 'trainBL.rds')
	saveRDS(yTrainBL, 'yTrainBL.rds')
	
	saveRDS(sosedSet, 'sosedSet.rds')
	saveRDS(ySosedSet, 'ySosedSet.rds')
	
	saveRDS(metaSet, 'metaSet.rds')
	saveRDS(yMetaSet, 'yMetaSet.rds')
	
	saveRDS(testSet, 'testSet.rds')
	saveRDS(yTestSet, 'yTestSet.rds')

C1 <- FM[class == 1,]
C2 <- FM[class == 2,]
plot(C1, col = 'red')
points(C2, col = 'blue', pch = '+')



for(i in 1:10){
	take <- createDataPartition(class, p = 0.1, list = FALSE)
	X <- FM[take,]
	y <- class[take]
	svp <- ksvm(X,y,type="C-svc")
	##plot(svp,data=FM)
		plot(scale(FM), col=class*2, pch=class, xlab="", ylab="")
		w <- colSums(coef(svp)[[1]] * FM[unlist(alphaindex(svp)),])
		b <- b(svp)
		abline(b/w[1],-w[2]/w[1], col = 'black', lwd = 2)
		abline((b+1)/w[1],-w[2]/w[1],lty=2)
		abline((b-1)/w[1],-w[2]/w[1],lty=2)
	ime <- paste0('ksvm', i, '.rds')
	setwd(OkoljeBL)
	saveRDS(svp, ime) 
	}



