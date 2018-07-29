### Data Preprocessing ###
library(data.table)
library(foreach)

# retrieve filenames of test sets
testfiles = list.files(pattern = "blogData_test")

# load and combine dataset
train = fread("blogData_train.csv")
test = foreach(i = 1:length(testfiles), .combine = rbind) %do% {
  temp = fread(testfiles[i], header = F)
}

# log-transform
train[, V281 := log(1 + V281)]
test[, V281 := log(1 + V281)]

# drop continous variables without variation
drop = c(8, 13, 28, 33, 38, 40, 43, 50, 278)
train[, (drop) := NULL]
test[, (drop) := NULL]

# write to files
write.csv(train, "BlogFeedback-Train.csv", row.names = F)
write.csv(test, "BlogFeedback-Test.csv", row.names = F)
