metaSet <- readRDS("C:\\Users\\Podlogar\\Documents\\DES\\Pima\\Rezultati\\CV1\\Osnovni podatki\\metaSet.rds")
sosedSet <- readRDS("C:\\Users\\Podlogar\\Documents\\DES\\Pima\\Rezultati\\CV1\\Osnovni podatki\\sosedSet.rds")

OP_meta <-  readRDS("C:\\Users\\Podlogar\\Documents\\DES\\Pima\\Rezultati\\CV1\\Napovedi BL\\OP_meta_verjetnost.rds")
OP_sosedi <-  readRDS("C:\\Users\\Podlogar\\Documents\\DES\\Pima\\Rezultati\\CV1\\Napovedi BL\\OP_sosedi_verjetnost.rds")


setwd("C:/Users/Podlogar/Documents/DES_ektimo/kNN")
write.table(metaSet, file = 'metaSet.csv', sep = ',', col.names = FALSE, row.names = FALSE)
write.table(sosedSet, file = 'sosedSet.csv', sep = ',', col.names = FALSE, row.names = FALSE)

write.table(OP_meta, file = 'OP_metaSet.csv', sep = ',', col.names = FALSE, row.names = FALSE)
write.table(OP_sosedi, file = 'OP_sosedSet.csv', sep = ',', col.names = FALSE, row.names = FALSE)


K  <- c(5, 10)
write.table(K, 'K.csv',col.names = FALSE, row.names = FALSE)

Kp <- c(5,10)
write.table(Kp, 'Kp.csv',col.names = FALSE, row.names = FALSE)

metric <- c('l2', 'chebyshev')
write.table(metric, 'metric.csv', col.names = FALSE, row.names = FALSE)


st <- Sys.time()
kNN_RC <- get.knnx(sosedSet, metaSet,  k=2)
matrika_sosediRC <- kNN_RC$nn.index
end <- Sys.time()
end - st

kNN_OP <- get.knnx(OPsosed, OPmeta,  k=2)
matrika_sosediOP <- kNN_OP$nn.index