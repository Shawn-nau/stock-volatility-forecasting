library(glmnet)
library(stringr)
library(highfrequency)
library(xts)
library(timetk)
library(dplyr)
library(ranger)

require(quantmod)
require(TTR)

## load data

## data could be found at link：https://pan.baidu.com/s/1hjqRyA1MUa9PHqw27zy5tg with code：9q5a 

stock_media<- local({ load('stock_media_300_paper_tech_sentiment.Rdat'); stock_media } )

############ feature sets

HAR_set<-c('rCov','rCov_lag1','rCov_mov7','rCov_mov30','Monday') 
online_news<- c('News_sentiment_lag1','Tnewsnum_lag1')
posts<- c('Post_sentiment_lag1','Tpostnum_lag1')
searching<- c('SVI_lag1')

online_news<- c('News_sentiment_lag1','Tnewsnum_lag1','News_sentiment_mov7','Tnewsnum_mov7')
posts<- c('Post_sentiment_lag1','Tpostnum_lag1','Post_sentiment_mov7','Tpostnum_mov7')
searching<- c('SVI_lag1','SVI_mov7')


HAR_online<- c(HAR_set,online_news)
HAR_posts<-c(HAR_set,posts)
HAR_searching<-c(HAR_set, searching)

HAR_media <-c(HAR_set, online_news, posts, searching)

Sent_s_vars<- c(HAR_set,'RSI_lag1','stochWPR_lag1','LTV_lag1','ABS_return_lag1','turnover_ratio_lag1','Swing_lag1')
Sent_e_vars<- c(HAR_set, 'RSI_A_lag1','stochWPR_A_lag1','LTV_A_lag1','ABS_return_A_lag1','ATR_A_lag1','SWING_A_lag1')
Sent_se_vars<- c(Sent_s_vars, 'RSI_mov7','stochWPR_mov7','LTV_mov7','ABS_return_mov7','turnover_ratio_mov7','Swing_mov7')

Media_vars0<-c(Sent_s_vars,online_news,posts,searching)  
Media_vars1<-c(Sent_se_vars,online_news,posts,searching) 

HAR_Sent_onlinenews<- c(Sent_e_vars, online_news)
HAR_Sent_posts<- c(Sent_e_vars,posts)
HAR_Sent_searching<- c(Sent_e_vars, searching)


summary_vars = c('rCov','RSI_lag1','stochWPR_lag1','LTV_lag1','ABS_return_lag1','turnover_ratio_lag1','Swing_lag1',
               'Tpostnum_lag1','Post_sentiment_lag1','Tnewsnum_lag1', 'News_sentiment_lag1','SVI_lag1')

S_cor= lapply(stock_media,function(x) {y=na.omit(x[,summary_vars]) ; 
									   cor(y)} ) 
S_cor =lapply(S_cor,function(x) {x[is.na(x)] = 0; x})

M_cor = S_cor[[1]]
for (i in 2:length(S_cor)) M_cor=M_cor+ S_cor[[i]]
M_cor = M_cor/300

write.table(M_cor, "clipboard", sep="\t", row.names=TRUE, col.names=TRUE) ## correlation matrix



########## roling forecasting with random forest
roling_step = 20
train_cal = seq(as.Date('2011-01-01'),as.Date('2020-01-01'),roling_step)
test_cal = seq(as.Date('2015-01-01'),as.Date('2020-01-01'),roling_step)


library(doParallel)
cl <- makeCluster(4)
registerDoParallel(cl)


info_set = Media_vars

rolling_sets<- lapply(1:length(stock_media),function(j){
		x <- stock_media[[j]]
		x$stock = names(stock_media)[j]
		x[,c(Media_vars1,'stock')]
	})
rolling_sets =  do.call(rbind,rolling_sets)


rf_preds = list()
for (info_set in list(HAR_set, HAR_online,HAR_posts,HAR_searching,HAR_media, Sent_se_vars, Media_vars1) ){

	rolling_preds <- foreach (i=1:length(test_cal),.packages=c('dplyr','xts','timetk','ranger')) %dopar% {

		train_set = paste0(train_cal[i],'/',test_cal[i]-1)
		test_set = paste0(test_cal[i],'/',test_cal[i]+roling_step-1)


		training_data = na.omit(rolling_sets[train_set,c(info_set,'stock')]) %>% tk_tbl() %>% select(-index)
		ranger.train <- ranger( rCov ~.,data = training_data)
		
		testing_data = rolling_sets[test_set,] #%>% tk_tbl() 
		x.na = complete.cases(testing_data[,info_set])
		preds = testing_data[,c('rCov','stock')] 
		preds[!x.na,'rCov']<-NA
		preds[x.na,'rCov'] = predict(ranger.train,data=testing_data[x.na,])$predictions
		
		rf_preds = lapply(names(stock_media),function(x) {index = preds[,'stock']==as.integer(x) ; preds[index,'rCov']}  )	
		rf_preds
	}

	rf_preds[[length(rf_preds)+1]] = rolling_preds

}
stopCluster(cl)

names(rf_preds) = c('HAR_set', 'HAR_online','HAR_posts','HAR_searching','HAR_media', 'Sent_se_vars', 'Media_vars1')


############################################## Rolling forecasting with Robust regression

library(doParallel)
cl <- makeCluster(8)
registerDoParallel(cl)
require ( MASS ) 

rlm_preds=list()

for (info_set in list(HAR_set, HAR_online,HAR_posts,HAR_searching,HAR_media, Sent_se_vars, Media_vars1) ){


	rolling_preds <- foreach (i=1:length(test_cal),.packages=c('dplyr','xts','timetk','MASS')) %dopar% {

		train_set = paste0(train_cal[i],'/',test_cal[i]-1)
		test_set = paste0(test_cal[i],'/',test_cal[i]+roling_step-1)

		rolling_sets<- lapply(1:length(stock_media),function(j){
			x <- stock_media[[j]]
			x$stock = names(stock_media)[j]
			x[train_set]
		})

		lm.news = lapply(rolling_sets, function(stock){
		
			stock = na.omit(stock[,info_set])
			model=NULL
			if (nrow(stock)>30){ 
				model = tryCatch({ rlm(rCov~ .,data=stock)},
				error= function(e){ lm(rCov~ rCov_lag1+rCov_mov7+rCov_mov30,data=stock)})
				}
			model
		} )
		
		stock_preds<- lapply(1:length(stock_media),function(j){
				x <- stock_media[[j]]
				m <- lm.news[[j]]
				if(!is.null(m)) preds= predict(m,newdata=x[test_set,])
				if(is.null(m)) preds= rep(NA,nrow(x[test_set,])); names(preds)= time(x[test_set,1])
				preds
			})	
		
		stock_preds
	}

	rlm_preds[[length(rlm_preds)+1]] = rolling_preds
}

stopCluster(cl)

names(rlm_preds) = c('HAR_set', 'HAR_online','HAR_posts','HAR_searching','HAR_media', 'Sent_se_vars', 'Media_vars1')


#################################################          evaluations  ###########################


roling_step = 20
train_cal = seq(as.Date('2011-01-01'),as.Date('2020-01-01'),roling_step)
test_cal = seq(as.Date('2015-01-01'),as.Date('2020-01-01'),roling_step)

roll_y_true = lapply(stock_media,function(x) x$rCov )
roll_y_true<- roll_y_true %>% do.call(cbind,.)
names(roll_y_true)<- names(stock_media)


##########  some utility functions for evaluation

preds_link <- function(preds_list){
	preds <- preds_list[[1]]
	for (i in 2:length(preds_list)) {
		temp = preds_list[[i]]
		for (j in 1:length(temp)) preds[[j]] = c(preds[[j]],temp[[j]])
		}
	preds	
}

preds_to_xts <- function(preds_list){

	lapply(preds_list,function(x)  data.frame(x,date= as.Date(names(x))) %>% tk_xts() )

}

evaluate_preds<- function(y_true,y_pred){

	err = y_true-y_pred
	ae = abs(err)
	se = err^2
	qlike = y_true/y_pred- log(y_true/y_pred)-1

	list(mae_time= rowMeans(ae,na.rm=TRUE),
		mae_stocks = colMeans(ae,na.rm=TRUE),
		me_stocks = colMeans(err,na.rm=TRUE),
		mse_time= rowMeans(se,na.rm=TRUE),
		mse_stocks = colMeans(se,na.rm=TRUE),
		ae=ae,
		se=se,
		qlike=qlike,
		qlike_stocks = colMeans(qlike,na.rm=TRUE)
	)

}

gmean<- function(x){ x %>% log %>% mean(.,na.rm=TRUE) %>% exp }

mcs_test<- function(loss_list){
	mcs = loss_list %>% do.call(cbind,.) %>% MCS::MCSprocedure(alpha=0.1,B=2000,statistic='Tmax')

	avg_loss = ((loss_list %>% do.call(cbind,.))/loss_list[[1]])  %>% apply(2,gmean)

	list(avg_loss,mcs@show %>% rownames)
}

modelnames = c('HAR', 'HAR+News','HAR + Guba','HAR+SVI','HAR+Online Sentiment', 'HAR+Technical Sentiment', 'HAR+Technical & Online Sentiment')

#### prepare baseline forecasts

baseline = rlm_preds[[1]] %>% preds_link()  %>% preds_to_xts() %>% do.call(cbind,.)
y_true<- roll_y_true[time(baseline)] #%>% exp
m0= baseline %>%  evaluate_preds(y_true,.)

#### rlm/rf overall evaluation

rlm_evaluate=list()

for (i in 1:length(rlm_preds)){
	preds = rlm_preds[[i]] %>% preds_link() %>% preds_to_xts() %>% do.call(cbind,.)
	mi= preds %>% evaluate_preds(y_true,.)
	rlm_evaluate = append(rlm_evaluate,list(mi$mse_stocks))
}

rlm_overall = mcs_evl(rlm_evaluate)



#### rlm/rf  yealy evaluation

###########


ep <- endpoints(m0$se,on="year") ## Locate the yearend
m0_year = period.apply(m0$se,INDEX=ep,FUN=mean) 

rlm_yearly_loss= list()

for (i in 1:length(rlm_preds)){

	preds = rlm_preds[[i]] %>% preds_link() %>% preds_to_xts() %>% do.call(cbind,.)

	mi= preds %>% evaluate_preds(y_true,.)
	mi_year = period.apply(mi$se,INDEX=ep,FUN=mean) 

	rlm_yearly_loss = append(rlm_yearly_loss,list(mi_year))

}


####### yearly mcs

list_trans<-function(x){
	z=list()
	n = length(x)
	m= nrow(x[[1]])
	for (i in 1:m) {
	    y=c()
		for (j in 1:n){
			y = rbind(y, x[[j]][i,])
		}
    z=append(z,list(t(y)))
	}
	z
	}
	
rlm_yearly_loss = list_trans(rlm_yearly_loss)

yearly_mcs = lapply(rlm_yearly_loss, function(x) {colnames(x) = modelnames; 
                 x= na.omit(x);
                 y = MCS::MCSprocedure(x,alpha=0.1,B=2000,statistic='Tmax');
				 y@show %>% rownames})
				 

