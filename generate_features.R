library('xcms')
library('plyr')
library('ggplot2')

data_name <- "data_name"  # name of the data
rt_range <- c(0, 1500)   # range for retention time
ppm<-20   #mass tolerence
cwp <- CentWaveParam(peakwidth = c(5, 100),snthresh=3,ppm=20)   #parameter for 'centWave' algorithm 
wd = '/../../../data_name'  #directory to store the target '.mzML' file
setwd(wd)
files <- list.files('.', '*.mzML')

raw_data <- readMSData(files = files, msLevel. = 1, mode = "onDisk")  
raw_data <- filterRt(raw_data, rt_range)

xsn <- findChromPeaks(raw_data, param = cwp)  #peak detection from raw data

peak_df <- data.frame(chromPeaks(xsn))   #save peak information as data.frame
peak_df <- peak_df[order(peak_df$mzmin), ]
peak_df$ppm <- .5 * 1e6 * (peak_df$mzmax - peak_df$mzmin) / peak_df$mz
peak_df$mzminexp <- pmin(peak_df$mz * (1 - ppm*1e-6), peak_df$mzmin)
peak_df$mzmaxexp <- pmax(peak_df$mz * (1 + ppm*1e-6), peak_df$mzmax)
peak_df$ppmexp <- 1e6 * (peak_df$mzmaxexp - peak_df$mzminexp) / peak_df$mz

peak_df <- peak_df[order(peak_df$mzminexp), ]

mz_range <- as.matrix(peak_df[, c("mzminexp", "mzmaxexp")])
dim(mz_range)

#################extract all the features from raw data and save as .txt files##############
start_time = Sys.time()
crs <- chromatogram(xsn, rt=rt_range, mz=mz_range, missing=0, aggregationFun='sum') # extract all EICs
end_time = Sys.time()
end_time - start_time

mzs <- c()
ints <- c()
rts <- rtime(crs[[1]])
for(j in 1:length(crs)){
  mzs <- c(mzs, mz(crs[[j]]))
  ints <- c(ints, intensity(crs[[j]]))
  cat("\r", "                             ")
  cat("\r",j)
}
mzs <- t(matrix(mzs, nrow = 2))
ints <- t(matrix(ints, nrow = length(rts)))
rts <- matrix(rts, ncol = 1)

write.table(ints, file = paste(c(data_name, 'ints.txt'), collapse = '_'), row.names = FALSE, col.names = FALSE, sep=",")
write.table(mzs, file = paste(c(data_name, 'mzs.txt'), collapse = '_'), row.names = FALSE, col.names = FALSE, sep=",")
write.table(rts, file = paste(c(data_name, 'rts.txt'), collapse = '_'), row.names = FALSE, col.names = FALSE, sep=",")
write.csv(peak_df, file = paste(c(data_name, 'peak_info.csv'), collapse = '_'))
