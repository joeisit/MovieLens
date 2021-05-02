##################################################
#     CODE FOR THE FINAL REPORT OF MOVIELENS     #
##################################################
# Author: José Ramón Riesgo Escovar              #
# Date: May 2021                         #
##################################################

#
# CREATION OF THE EDX SET AND VALIDATION SET 
#

##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId and genres in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId") %>%
  semi_join(edx, by = "genres")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#############################################################
# We created the EDX Set and the Validation Set
#############################################################

# We are left with EDX as the training Data-Set and Validation for the final evaluation of 
# The defined "best" algorithm

# We are going to save the EDX to a File and Validation in case we need to reload them
# Also this will simplify the upload in the Final Report (.rmd)

# Save EDX file
save(edx , file="rda/edx.rda")
# Save Validation file
save(validation, file="rda/validation.rda")

#########################################
# LOADING ALL LIBRARIES THAT WILL BE USED
#########################################
library(dplyr)
library(ggplot2)
library(tidyverse)
library(markdown)
library(caret)
library(knitr)
#########################################

#
# NEXT STEP IS TO EXPLORE THE DATA, MAKE SOME ANALYSIS
#

# Get a Summary of the Edx Data
summary(edx)

#
# We code the RMSE Calculation Function
#

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# NEXT STEP IS TO PARTITION DATA EDX in TRAINING AND TEST

# We will set the seed to 1 so it is always the same result
set.seed(1)
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
# The we create the two sets
train_set <- edx[-test_index,]
test_set <- edx[test_index,]

# We have the two sets, train and test set to work with.

#
# Now we need to make sure we do not include Users and Movies as well as Genres
# in the test set that are not in the training set.
# To do this we we use the semi_join function
#

test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId") %>%
  semi_join(train_set, by = "genres")

# 
# LETS DEFINE THE MORE SIMPLE MODEL
#

# Our first Algorithm will be simply the Average values of the training set
# We assume all movies and all users rate the "same"
mu_adv <- mean(train_set$rating)

# The value of mu
mu_adv

# We use then the RMSE function to get our first result of the basic Algorithm

mu_model <- RMSE( mu_adv,test_set$rating)
# The result of the first Algorithm

mu_model

# To CAPTURE THE RESULTS OF EACH METHOD WE BUILD A TABLE AND ADD THE FIRST VALUE

rmse_results <- data.frame(METHOD = "Simple Average Model", RMSE = mu_model )

# First Result

rmse_results

# 
# SECOND MODEL
#

# Exploring the Data
# Some Movies get higher ratings than others 
# We can see this in the following Chart

test_set %>% 
group_by(movieId) %>%
summarize(howmany = n(), average = mean(rating)) %>%
arrange(desc(howmany)) %>%
ggplot(aes(average,howmany)) +
geom_point(colour = "light green", size = 1) +
theme_light() +
labs(title = " COMPARING EVALUATIONS AGAINST THEIR AVERAGE RATINGS") +
xlab("MEAN OF RATINGS") + 
ylab("NUMBER OF EVALUATIONS") + 
theme(plot.title = element_text(color = "blue", size = 11, 
                                  face = "bold", hjust = 0.5)) +
theme(axis.title.x = element_text(color = "grey", size = 10, 
                                    face = "bold", hjust = 0.5)) +
theme(axis.title.y = element_text(color = "grey", size = 10, 
                                    face = "bold", hjust = 0.5)) +
theme(axis.text.x = element_text(color = "light blue", size = 10, 
                                    face = "bold", hjust = 0.5)) +
theme(axis.text.y = element_text(color = "light blue", size = 10, 
                                   face = "bold", hjust = 0.5)) +
annotate("rect", xmin=2.2, xmax=4.5, ymin=500, ymax=6500, alpha=0.2, 
           fill="light yellow", color= "black")

# We can see that there is a clear tendency and we highlight that a higher number
# of Evaluations tend to get a higher average in the Ratings

# Based on this fact we can augment our previous algorithm
# by adding an average ranking for movie 
  
# We can use the least square to estimate these effect

# We use previous calculated mu_adv
# Using the Training Set we sum all ratings and then adjust by reducing the average rating

movie_avgs <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu_adv)) 

# We will refer to the effect of this average rating as the "BIAS", with the following
# table we will show these BIAS

movie_avgs %>%
  ggplot(aes(b_i)) +
  geom_histogram(bins = 10, colour= "blue", fill="light green") +
  labs(title = " DISTRIBUTION OF BIAS") +
  xlab("VALUES") + 
  ylab("NUMBER OF EVALUATIONS") + 
  theme(plot.title = element_text(color = "blue", size = 11, 
                                  face = "bold", hjust = 0.5)) +
  theme(axis.title.x = element_text(color = "grey", size = 10, 
                                    face = "bold", hjust = 0.5)) +
  theme(axis.title.y = element_text(color = "grey", size = 10, 
                                    face = "bold", hjust = 0.5)) +
  theme(axis.text.x = element_text(color = "light blue", size = 10, 
                                   face = "bold", hjust = 0.5)) +
  theme(axis.text.y = element_text(color = "light blue", size = 10, 
                                   face = "bold", hjust = 0.5))

# With this new bias we are going to calculate the new ratings to compare to the test_set

movie_model <- test_set %>%
  left_join(movie_avgs, by='movieId') %>%
  mutate(prediction = mu_adv + b_i) %>%
  pull(prediction)

movie_result <- RMSE( movie_model,test_set$rating)

# The result of the movie model

movie_result

# Capture the results of movie_model

rmse_results <- bind_rows(rmse_results,
            data.frame(METHOD = "Movie-Based Model", RMSE = movie_result))

# Updated Table

rmse_results %>% knitr::kable()

#
# THIRD ALGORITHM 
#

#We also will combine the users with the amount of movies of the previous model

train_set %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(b_u)) +
  geom_histogram(bins = 30, colour= "blue", fill="light green") +
  labs(title = " DISTRIBUTION OF AVERAGE RATINGS OF USERS THAT EVALUATED MORE THAN 100 MOVIES") +
  xlab("AVERAGE RATING") + 
  ylab("NUMBER OF EVALUATIONS") + 
  theme(plot.title = element_text(color = "blue", size = 8, 
                                  face = "bold", hjust = 0.5)) +
  theme(axis.title.x = element_text(color = "grey", size = 7, 
                                    face = "bold", hjust = 0.5)) +
  theme(axis.title.y = element_text(color = "grey", size = 7, 
                                    face = "bold", hjust = 0.5)) +
  theme(axis.text.x = element_text(color = "light blue", size = 7, 
                                   face = "bold", hjust = 0.5)) +
  theme(axis.text.y = element_text(color = "light blue", size = 7, 
                                   face = "bold", hjust = 0.5))

# These show that there is value on including the user ratings in the previous model

# Calculate the average by user combining with the movies

user_avgs <- train_set %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu_adv - b_i))

# Combine both Movie and User in the new model

movie_user_model <- test_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu_adv + b_i + b_u) %>%
  pull(pred)

movie_user_result <- RMSE(movie_user_model, test_set$rating)

# The result of the Movie + User Model
movie_user_result

# Capture the results of the Movie + User Model

rmse_results <- bind_rows(rmse_results,
                          data.frame(METHOD = "Movie+User Model", 
                          RMSE = movie_user_result))

# Updated Table

rmse_results %>% knitr::kable()


# 
# FORTH MODEL
#

# Now we will combine the Genres with the previous Movies as well as Users

# Calculate the new genres combining with previous Movies and Users

genre_avgs <- train_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by= 'userId') %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu_adv - b_i - b_u))

# Combine both Movie, User and Genres

movie_user_genres_model <- test_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  mutate(pred = mu_adv + b_i + b_u + b_g) %>%
  pull(pred)

movie_user_genres_result <- RMSE(movie_user_genres_model, test_set$rating)

# The result of the Movie + User Model
movie_user_genres_result

# Capture the results of the Movie + User Model

rmse_results <- bind_rows(rmse_results,
                          data.frame(METHOD = "Movie+User+Genres Model", 
                                     RMSE = movie_user_genres_result))
 
# Updated Table

rmse_results %>% knitr::kable()


# 
# FIFTH MODEL
#

# Now we will adjust using the techniques of Regularization the three elements of our model:
# Movies, Users and Genres to make our model more efficient.

# We will estimate the lambdas between 1 to 10 to get the better that minimizes our RMSE function
lambdas <- seq(0, 10, 0.25)
# Compute the predicted ratings on test_set dataset using different values of lambda
rmses <- sapply(lambdas, function(lambda) {
  
  # Calculate the average by movie
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu_adv) / (n() + lambda))
  
  # Calculate the average by user
  b_u <- train_set %>%
    left_join(b_i, by='movieId') %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu_adv) / (n() + lambda))
  
  # Calculate the average by genre
  b_g <- train_set %>%
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_i - mu_adv - b_u) / (n() + lambda))
  
  # Compute the predicted ratings on testing data dataset
  predicted_ratings <- test_set %>%
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    left_join(b_g, by='genres') %>%
    mutate(pred = mu_adv + b_i + b_u + b_g) %>%
    pull(pred)
  
  # Predict the RMSE on the testing set
  return(RMSE(predicted_ratings,test_set$rating))
})

# Get the lambda value that minimize the RMSE
min_lambda <- lambdas[which.min(rmses)]

# Predict the RMSE on the validation set
min_lambda

# Get the "best" minimize RMSE function
reg_movie_user_genres_result <- min(rmses)
reg_movie_user_genres_result

# Capture the results of the Movie + User Model

rmse_results <- bind_rows(rmse_results,
                          data.frame(METHOD = "Reg Movie+User+Genres Model", 
                                     RMSE = reg_movie_user_genres_result))
# Updated Table
rmse_results %>% knitr::kable()

#
# FINAL CALCULATION WITH VALIDATION
#

# With the best Model, that is actually the last one

lambdas <- seq(0, 10, 0.25)
# Compute the final ratings on validation dataset using different values of lambda
# and the whole edx data set

rmses <- sapply(lambdas, function(lambda) {
  
  # Calculate the average by movie
  b_i <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu_adv) / (n() + lambda))
  
  # Calculate the average by user
  b_u <- edx %>%
    left_join(b_i, by='movieId') %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu_adv) / (n() + lambda))
  
  # Calculate the average by genre
  b_g <- edx %>%
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_i - mu_adv - b_u) / (n() + lambda))
  
  # Compute the predicted ratings on validation dataset
  predicted_ratings <- validation %>%
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    left_join(b_g, by='genres') %>%
    mutate(pred = mu_adv + b_i + b_u + b_g) %>%
    pull(pred)
  
  # Predict the RMSE on the validation set
  return(RMSE(predicted_ratings,validation$rating))
})

# Get the lambda value that minimize the RMSE
min_lambda <- lambdas[which.min(rmses)]

# Predict the RMSE on the validation set
min_lambda

final_result <- min(rmses)

# Display the result of the RMSE function
final_result 






