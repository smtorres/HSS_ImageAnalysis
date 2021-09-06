rm(list=ls())
library(readxl)

# Import traits data
setwd("Data") # Folder where the full 10K US Adult Faces Database Resources folder is saved
dat <- read_excel("10K US Adult Faces Database Resources/Full Attribute Scores/psychology attributes/psychology-attributes.xlsx", sheet = 1)
# The file is in the folder Full Attribute Scores > psychology attributes > psychology-attributes.xlsx

# Get mean scores by picture
aggrdat <- aggregate.data.frame(dat[,c("happy")], by=list(dat$Filename), FUN = mean)
head(aggrdat)
colnames(aggrdat)[1] <- "Filename"

# Some basic clean-up
table(aggrdat$happy)
aggrdat$happy2 <- round(aggrdat$happy)
table(aggrdat$happy2)
aggrdat$happy2 <- aggrdat$happy2-1
table(aggrdat$happy2)

# Split training and testing data
set.seed(2901)
index_sample <- sample(1:nrow(dat), 3000)


train <- dat[index_sample[1:2000],]
test <- dat[index_sample[2001:3000],]
head(train)

# Transfer selected images to a designated folder
img_folder <- "10K US Adult Faces Database Resources/10k US Adult Faces Database/Face Images/"
new_folder <- "Face Images/" # Folder to store the training and testing images [the folder will be uploaded as such to Google Drive]

index_errors <- file.copy(paste0(img_folder, dat$Filename[index_sample]),new_folder)

# Some cannot be copied 
train <- train[index_errors[1:2000],]
test <- test[index_errors[2001:3000],]


# Sanity checks
check0 <- sapply(train$Filename, function(x) x%in% list.files(new_folder))
sum(check0)==nrow(train)
check1 <- sapply(test$Filename, function(x) x%in% list.files(new_folder))
sum(check1)==nrow(test)

# Save the data
write.csv(train, file="PsychAttr_train.csv", row.names = FALSE)
write.csv(test, file="PsychAttr_test.csv", row.names = FALSE)

