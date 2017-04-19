args <- commandArgs(trailingOnly = TRUE)
in_file <- args[1]

library(jpeg)
img <- readJPEG(in_file)
imgDm <- dim(img)

imgRGB <- data.frame(
    x = rep(1:imgDm[2], each = imgDm[1]),
    y = rep(imgDm[1]:1, imgDm[2]),
    R = as.vector(img[,,1]),
    G = as.vector(img[,,2]),
    B = as.vector(img[,,3])
    )

kClusters <- args[2]

ptm <- proc.time()
kMeans <- kmeans(imgRGB[, c("R", "G", "B")], centers = kClusters)
kColors <- rgb(kMeans$centers[kMeans$cluster,])
elapsed <- proc.time() - ptm
print(elapsed)

library(ggplot2)

jpeg(paste(paste('R_k=',kClusters), in_file, sep='_'), width=imgDm[2], height=imgDm[1])

ggplot(data = imgRGB, aes(x = x, y = y)) +
  geom_point(colour = kColors) +
  scale_x_continuous(expand=c(0,0)) +
  scale_y_continuous(expand=c(0,0)) +
  theme_void() + labs(x = NULL, y = NULL)
