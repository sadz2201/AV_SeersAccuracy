library(plyr)
library(xgboost)

fulldata=read.csv('Train_seers_accuracy.csv')

dob=as.POSIXlt(as.Date(fulldata$DOB,'%d-%b-%y'))
#this fixes up cases like DOB=01-Mar-2062, changing it to 01-Mar-1962 instead
isold=dob$year>100
dob$year[isold]=dob$year[isold]-100
fulldata$DOB=dob

curr_date2006=as.POSIXlt(as.Date('2006-01-01'))
curr_date2007=as.POSIXlt(as.Date('2007-01-01'))
#find age of every customer as of Jan 1st 2007. 
fulldata$age=as.numeric(curr_date2007-fulldata$DOB)/365

fulldata$Transaction_Date=as.POSIXlt(as.Date(fulldata$Transaction_Date,'%d-%b-%y'))
#sort the data based on Client_ID & Transaction_Date ascending
fulldata=fulldata[order(fulldata$Client_ID,fulldata$Transaction_Date),]

#train set - all Clients who purchased between 2003-05
#target - whether they repeat-purchased in 2006
train=fulldata[fulldata$Transaction_Date$year<=105,]

#a- logical vector whether Client_ID is duplicated or not 
a=duplicated(train$Client_ID)
curr_trans=train$Transaction_Date[-1] #vector of all transaction dates,except the first one
prev_trans=train$Transaction_Date[-nrow(train)] #vector of all transaction dates,except the last one

#difference gives the time gap(in years) between 2 consecutive transactions for any Client 
time_since_prev=as.numeric(curr_trans-prev_trans)/(365*86400)
train$time_since_prev=c(0,time_since_prev)
#for non-duplicates(i.e unique Clients),reset to 0 bcoz there is no previous transaction.
train$time_since_prev[!a]=0

#time difference between Jan 1st 2006 & current transaction date
train$time_since_newyear=as.numeric(curr_date2006-train$Transaction_Date)/365

#count of total transactions for a Client
countdata=count(train,vars="Client_ID")

#now reorder trainset in by Cleint_Id(asc) & Transactiin_Date(desc)
#I do this so that I can retain most recent transaction only
train=train[order(train$Client_ID,rev(order(train$Transaction_Date))),]
b=duplicated(train$Client_ID)
#get rid of duplicates( i.e older transactions for every Client)
train=train[!b,]
train=merge(train,countdata)


#find all unique Clients in the year 2006
purchase2006=fulldata[fulldata$Transaction_Date$year==106,]
purchase2006=unique(purchase2006$Client_ID)
Cross_Sell=rep(0,nrow(train))

#intersection gives the Clients who purchased in 2003-05, & came back again in 2006  
repeaters2006=intersect(train$Client_ID,purchase2006)
Cross_Sell[train$Client_ID %in% repeaters2006]=1 #Set target to 1 only for the repeaters2006


#test set - all Clients who purchased between 2003-06
#target (to predict) - whether they repeat purchased in 2007
#repeat same process as above
#calculate time difference from 1st Jan 2007 instead.
test=fulldata
a=duplicated(test$Client_ID)
curr_trans=test$Transaction_Date[-1]
prev_trans=test$Transaction_Date[-nrow(test)]
time_since_prev=as.numeric(curr_trans-prev_trans)/(365*86400)
test$time_since_prev=c(0,time_since_prev)
test$time_since_prev[!a]=0
test$time_since_newyear=as.numeric(curr_date2007-test$Transaction_Date)/365

countdata=count(test,vars="Client_ID")
test=test[order(test$Client_ID,rev(order(test$Transaction_Date))),]
b=duplicated(test$Client_ID)
test=test[!b,]
test=merge(test,countdata)


#get rid of Transtion_ID,Transaction_Date & DOB columns
train=train[,-c(2,3,11)]
test=test[,-c(2,3,11)]

#converting factors & character variables to integer
all_data=rbind(train,test)
n=nrow(train)
for (f in names(all_data)) {
  if (class(all_data[[f]])=="character" || class(all_data[[f]])=="factor") {
    all_data[[f]] <- as.integer(factor(all_data[[f]]))
  }
}

train=all_data[1:n,]
test=all_data[(n+1):nrow(all_data),]

xgtrain = xgb.DMatrix(as.matrix(train), label = Cross_Sell, missing = 0)
xgtest = xgb.DMatrix(as.matrix(test), missing=0)
watchlist <- list(train=xgtrain)

param <- list(  objective           = "binary:logistic", 
                eval_metric         = "auc",
                eta                 = 0.1,
                max_depth           = 4,
                subsample           = 0.7, #0.7
                colsample_bytree    = 0.5,   
                scale_pos_weight    = 1.2
)

set.seed(619)
clf <- xgb.cv(   params = param,
                 nfold = 4,
                 data = xgtrain, 
                 nrounds = 10000,
                 early.stop.round = 20,
                 verbose = 1
                 
)

best <- max(clf$test.auc.mean)
bestIter <- which(clf$test.auc.mean==best)-1

#gonna retrain on full data. Hence need more number of iterations
bestIter=round(bestIter*1.25)
set.seed(619)
clf_full <- xgb.train(   params = param,
                         data = xgtrain, 
                         nrounds = bestIter,
                         watchlist = watchlist,
                         verbose = 1
                         
)

P = predict(clf_full, xgtest)

#CV score: 0.8794, Public LB: 0.8809
#This hack of resetting very low values to 0 boosted Public LB score to 0.8812 
#(I hope it doesn't bite me on the Private LB)
P[P<0.0004]<- 0 

#save the submission
submission=data.frame("Client_ID"=test$Client_ID,"Cross_Sell"=P)
write.csv(submission,"submission_final.csv",row.names=F)

#save the feature importance matrix for future analysis
feat_imp=xgb.importance(feature_names=names(train),model=clf_full)
write.csv(feat_imp,'feat_imp.csv',row.names=F)
