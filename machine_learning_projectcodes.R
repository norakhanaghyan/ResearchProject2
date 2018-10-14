
library(readr)
library(stringr)
library(plyr)
library(tree)
library(ggplot2)
library(corrplot)
library(pca3d)
library(car)
library(dplyr)
library(glmnet)
library(useful)
library(coefplot)
library(tree)
library(randomForest)
library(gbm)
library(dummies)
library(sjPlot)
movies<-read_csv('C:/Users/Norvard/Desktop/movies.csv')
movies<-na.omit(movies)

movies$genres<-word(movies$genres, 1, sep=fixed('|'))
table(movies$genres)
movies$genres <- mapvalues(movies$genres, from=c("Romance",
                                                 "Adventure", "Western", "Thriller", "Sci-Fi", 
                                                 "Musical", "Family",
                                                 "Documentary"), 
                           to=c("Drama", 
                                "Action", "Action", "Action", "Action", 
                                "Comedy", "Comedy", 
                                "Biography"))


table(movies$genres)
movies$genres<-as.factor(movies$genres)
movies<- cbind(movies, dummy(movies$genres, sep = "_"))
movies$title_year<-as.factor(movies$title_year)
movies$color<-as.factor(movies$color)

# Descriptive statistics
summary(movies)
ggplot(movies, aes(x=imdb_score))+geom_histogram(fill='light blue')+ggtitle('IMDB_score distribution')
ggplot(movies, aes(x=title_year, y=imdb_score))+geom_boxplot(fill='light blue')+theme(axis.text.x = element_text(angle = 90, hjust = 1))+ggtitle('IMDB vs Year')
ggplot(movies, aes(x=country, y=imdb_score))+geom_boxplot(fill='light blue')+theme(axis.text.x = element_text(angle = 90, hjust = 1))+ ggtitle('IMDB vs Country')
ggplot(movies, aes(x=language, y=imdb_score))+geom_boxplot(fill='light blue')+theme(axis.text.x = element_text(angle = 90, hjust = 1))+ ggtitle('IMDB vs Language')
ggplot(movies, aes(x=imdb_score, y=movie_facebook_likes))+geom_point(color='light blue')+ ggtitle('IMDB vs Facebook popularity of the movie')
ggplot(movies, aes(x=imdb_score, y=director_facebook_likes))+geom_line(color='light blue')+geom_smooth()+ ggtitle('IMDB vs Facebook popularity of the director')

# Finding correlations

movies_cont<- movies[, c(24, 26:34, 3:6, 8,9,13,14,17, 21, 23, 25)]
  
cor(movies_cont)
corrplot(cor(movies_cont), method='ellipse')

reduced_movies<-movies[, c(24, 26:28, 30:34, 4:6, 8,9, 21, 23, 25)]
sort(cor(reduced_movies))
corrplot(cor(reduced_movies), method='ellipse')

reduced_movies_final<-movies[, c(24, 26:28, 30:34, 4:6, 8, 21, 23, 1)]


# Multiple Linear regression

full<-lm(imdb_score~., reduced_movies_final)
summary(full)
AIC(full)
BIC(full)

# stepwise regression
empty = lm(imdb_score ~ 1, reduced_movies_final) #The model with an intercept ONLY.
scope = list(lower = formula(empty), upper = formula(full))
forwardAIC = step(empty, scope, direction = "forward", k = 2)

summary(forwardAIC)
vif(forwardAIC)

sjt.lm(empty, full, forwardAIC, group.pred = FALSE)

# splitting data
set.seed(1)
index<-sample( 1: nrow(movies), 0.8*nrow(movies))
movies.train<-movies[index,]
movies.test<-movies[-index, ]
movies.train<- movies.train%>% mutate_if(is.character,as.factor)
movies.test<- movies.test%>% mutate_if(is.character,as.factor)
movies.train<- movies.train%>% mutate_if(is.integer,as.numeric)
movies.test<- movies.test%>% mutate_if(is.integer,as.numeric)


formula<-imdb_score~color+duration+actor_3_facebook_likes+actor_1_facebook_likes+actor_2_facebook_likes +
budget+director_facebook_likes+movies_Action+movies_Animation+movies_Biography+movies_Crime+
movies_Drama+movies_Fantasy+movies_Horror+movies_Mystery-1

input_x_train<-build.x(formula, data=movies.train, contrasts=FALSE, sparse=TRUE)
output_y_train<-build.y(formula, movies.train)
input_x_test<-build.x(formula, data=movies.test, contrasts=FALSE, sparse=TRUE)
output_y_test<- build.y(formula, movies.test)

#Ridge
ridge<-cv.glmnet(x=input_x_train, y=output_y_train, family='gaussian', alpha=0)
ridge.mod<-glmnet(x=input_x_train, y=output_y_train, family='gaussian', alpha=0)
best_lambda1<-ridge$lambda.min
coef(ridge)
coefpath(ridge)
coefplot(ridge)
plot(ridge.mod, xvar='lambda')
ridge.predict<-predict(ridge, s=best_lambda1, newx<-input_x_test)
mse_ridge<-mean((ridge.predict-output_y_test)^2)
mse_ridge

# LAsso
lasso<-glmnet(x=input_x_train, y=output_y_train, family='gaussian')
plot(lasso, xvar = 'lambda')
#cross_validation
validation<-cv.glmnet(x=input_x_train, y=output_y_train, family='gaussian', nfolds = 5)
best_lambda<-validation$lambda.min
plot(validation)
coefplot(validation, sort='mag', lambda='lambda.min')
coefpath(lasso)
coef(lasso)

lasso1<-glmnet(x=input_x_train, y=output_y_train, family='gaussian', lambda=validation$lambda.min)
coefplot(lasso1)
coef(lasso1)
lasso_predict<-predict(lasso1, s=best_lambda, newx=input_x_test)
mse_lasso<-mean((lasso_predict-output_y_test)^2)
mse_lasso


# Tree
formula1<-imdb_score~color+duration+actor_3_facebook_likes+actor_1_facebook_likes+actor_2_facebook_likes +
budget+director_facebook_likes+movies_Action+movies_Animation+movies_Biography+movies_Crime+
movies_Drama+movies_Fantasy+movies_Horror+movies_Mystery
tree.movies <- tree(formula1, movies.train, split='deviance')
tree.movies1 <- tree(formula1, movies.train, split='gini',
                   control=
                     tree.control(nobs=nrow(movies.train), 
                                  mincut=20, minsize=40, mindev=0.01))
summary(tree.movies)
summary(tree.movies1)
plot(tree.movies)
title(main='Splitting_Deviance')
text(tree.movies, pretty = 1, cex = .5)
plot(tree.movies1)
title(main='Splitting_Gini')
text(tree.movies1, pretty = 0, cex = .5)

##prediction
tree.pred <- data.frame(predict(tree.movies,movies.test))
mse_tree<-mean((tree.pred-movies.test$imdb_score)^2)
mse_tree

# RandomForest

movies_randomForest<-randomForest(formula1, data = movies.train, mtry = 4, importance = TRUE)
varImpPlot(movies_randomForest)
movies_randomForest
plot(movies_randomForest)

## prediction
pred.movies_rf<- data.frame(predict(movies_randomForest,newdata=movies.test))
mse_rf<-mean((pred.movies_rf-movies.test$imdb_score)^2)
mse_rf

#Boosting

movies_boost <- gbm(formula1, data = movies.train, n.trees = 5000, interaction.depth=4)
summary(movies_boost)

## prediction
pred.movies_boost <- data.frame(predict(movies_boost, newdata = movies.test, n.trees = 5000))
mse_boost<-mean((pred.movies_boost-movies.test$imdb_score)^2)
mse_boost

results = data.frame(Model = c("Single Tree", "Lasso",  "Ridge","RandomForest",  "Boosting"),
                     TestError = c(mse_tree, mse_lasso, mse_ridge, mse_rf, mse_boost))

df_list<-data.frame(pred.movies_rf, pred.movies_boost)
df_list$ave_pred<-rowMeans(df_list)

mse_av_pred<-mean((df_list$ave_pred-movies.test$imdb_score)^2)
mse_av_pred
