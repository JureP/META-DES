## http://danielnee.com/tag/platt-scaling/

#####
OkoljePodatki <- 'Z:/JurePodlogar/Ensemble selection/Podatki/Popravljeni'
OkoljeSave <- 'Z:/JurePodlogar/Ensemble selection/Selection/Scaling'
setwd(OkoljePodatki)
podatki <- readRDS('razdelitevPodatkov_list.rds')

X <- podatki$xTrain
#Y <- as.numeric(podatki$yTrain)
Y <- podatki$yTrain

	
	
Xscale <- 0 * X
names(Xscale) <- names(X)
	
for (kateri in 1:ncol(X)){
	#reliability.plot(Y, X[,kateri])

	Ycalib <- Y
	Xcalib <- X[ , kateri]
				
	calibDataFrame <- data.frame(cbind(Ycalib, Xcalib))
	colnames(calibDataFrame) <- c("y", "x")
	calibModel <- glm(y ~ x, calibDataFrame, family=binomial)
	calibDataFrame <- data.frame(Xcalib)
	colnames(calibDataFrame) <- c("x")
	Xcalibrated <- predict(calibModel, newdata=calibDataFrame, type="response")
	Xscale[, kateri] <- Xcalibrated
	
	# par(mfrow = c(2,1))
	 hist(Xcalibrated, main = mean(Xcalibrated))
	# hist(X[, kateri], main = mean(X[, kateri]))
	# title(names(X)[kateri], outer=TRUE)
}
	
setwd(OkoljeSave)
saveRDS(Xscale, 'XtrainScaled.rds')



reliability.plot <- function(obs, pred, bins=10, scale=T) {
  #  Plots a reliability chart and histogram of a set of predicitons from a classifier
  #
  # Args:
  #   obs: Vector of true labels. Should be binary (0 or 1)
  #   pred: Vector of predictions of each observation from the classifier. Should be real
  #       number
  #   bins: The number of bins to use in the reliability plot
  #   scale: Scale the pred to be between 0 and 1 before creating reliability plot
  require(plyr)
  library(Hmisc)

  min.pred <- min(pred)
  max.pred <- max(pred)
  min.max.diff <- max.pred - min.pred
  
  if (scale) {
    pred <- (pred - min.pred) / min.max.diff 
  }
  
  bin.pred <- cut(pred, bins)
  
  k <- ldply(levels(bin.pred), function(x) {
    idx <- x == bin.pred
    c(sum(obs[idx]) / length(obs[idx]), mean(pred[idx]))
  })
  
  is.nan.idx <- !is.nan(k$V2)
  k <- k[is.nan.idx,]  
  plot(k$V2, k$V1, xlim=c(0,1), ylim=c(0,1), xlab="Mean Prediction", ylab="Observed Fraction", col="red", type="o", main="Reliability Plot")
  lines(c(0,1),c(0,1), col="grey")
  subplot(hist(pred, xlab="", ylab="", main="", xlim=c(0,1), col="blue"), grconvertX(c(.8, 1), "npc"), grconvertY(c(0.08, .25), "npc"))
}

i <- 10
reliability.plot(as.numeric(Y), X[,i], 10, F)
reliability.plot(as.numeric(Y), Xscale[,i], 10, F)
