# Chapter 10 - Nonlinear regression with generalized additive models

#### Your job is to build a regression model that can predict ozone pollution levels based on the time of year and meteorological readings, such as humidity and tem- perature.

##Building Regression Model

library(mlr)

library(tidyverse)

library(mlbench)

data(Ozone)

ozoneTib <- as_tibble(Ozone)

names(ozoneTib) <- c("Month", "Date", "Day", "Ozone", "Press_height",
                     "Wind", "Humid", "Temp_Sand", "Temp_Monte",
                     "Inv_height", "Press_grad", "Inv_temp", "Visib")

ozoneTib

####Converting all column classes to numeric. And Since we are trying to predict Ozone, to prevent any bias, piping into filter NA is false

ozoneClean <- mutate_all(ozoneTib, as.numeric) %>%
  filter(is.na(Ozone) == FALSE)

ozoneClean

## Making linear regression nonlinear with polynomial terms

####For a single predictor variable, we can generalize this for any nth-degree polynomial relationship as:
#### y=β0 +β1x+β1x2 +...βnxn +ε

## More flexibility: Splines and generalized additive models

####A spline is a piecewise polynomial function. This means it splits the predictor variable into regions and fits a separate polynomial within each region, which regions connect to each other via knots. 
####A knot is a position along the predictor variable that divides the regions within which the separate polynomials are fit. The polynomial curves in each region of the predictor pass through the knots that delimit that region. This allows us to model complex nonlinear relationships that are not constant across the range of the predictor variable.
#### But this approach has some limitations:
#### The position and number of the knots need to be chosen manually. Both choices can make a big impact on the shape of the spline. The choice of knot position is typically either at obvious regions of change in the data or at regular intervals across the predictor, such as at the quartiles.
#### The degree of the polynomials between knots needs to be chosen. We generally use cubic splines or higher, because these ensure that the polynomials connect with each other smoothly through the knots (quadratic polynomials may leave the spline disconnected at the knots).
#### It can become difficult to combine splines of different predictors.

#### The Solution for this issue is GAM:
#### GAM take form: y=β0 +f1(x1)+f2(x2)+...fk(xk)+ε ; instead of y=β0 +β1x+β2x2 +...β2x2 +ε
#### where each f(x) represents a function of a particular predictor variable. These func- tions can be any sort of smoothing function but will typically be a combination of mul- tiple splines.

### How GAMs learn their smoothing functions
####The most common method of constructing these smoothing functions is to use splines as basis functions. Basis functions are simple functions that can be combined to form a more complex function.
####The function fk(xk) can be expressed as: f(xi) = a1b1(xi) + a2b2(xi) + ... + anbn(xi)
####where b1(xi) is the value of the first basis function evaluated at a particular value of x, and a1 is the weight of the first basis function. GAMs estimate the weights of these basis functions in order to minimize the residual square error of the model.

#### GAMs automatically learn a nonlinear relationship between each predictor vari- able and the outcome variable, and then add these effects together linearly, along with the intercept. GAMs overcome the limitations of simply using splines in the gen- eral linear model by doing the following:
####  Automatically selecting the knots for spline functions
####  Automatically selecting the degree of flexibility of the smoothing functions by controlling the weights of the basis functions
####  Allowing us to combine splines of multiple predictor variables simultaneously

### How GAMs handle categorical variables

#### GAMs can handle categorical variables in two different ways.
#### One method is to treat categorical variables exactly the same way we do for the general linear model, and create k – 1 dummy variables that encode the effect of each level of the predictor on the outcome. When we use this method, the predicted value of a case is simply the sum of all of the smoothing functions, plus the contribution from the categorical variable effects. This method assumes independence between the categorical variable and the continuous variables (in other words, the smoothing functions are the same across each level of the categorical variable).
#### The other method is to model a separate smoothing function for each level of the categorical variable. This is important in situations where there are distinct nonlinear relationships between continuous variables and the outcome at each level of a cate- gorical variable.

## Building GAM:

####Let’s get day-of-the-year resolution from our data. To achieve this, we mutate a new column called DayOfYear. We use the interac- tion() function to generate a variable that contains the information from both the Date and Month variables. Because the interaction() function returns a factor, we wrap it inside the as.numeric() function to convert it into a numeric vector that rep- resents the days of the year.
####Because the new variable contains the information from the Date and Month variables, we remove them from the data using the select() function—they are now redun- dant. We then plot our new variable to see how it relates to Ozone.

ozoneForGam <- mutate(ozoneClean,
                      DayOfYear = as.numeric(interaction(Date, Month))) %>%
  select(c(-"Date", -"Month"))

ggplot(ozoneForGam, aes(DayOfYear, Ozone))  +
  geom_point()  +
  geom_smooth() +
  theme_bw()

####Now let’s define our task, imputation wrapper, and feature-selection wrapper, just as we did for our linear regression model. Sadly, there isn’t yet an implementation of ordinary GAMs wrapped by mlr (such as from the mgcv package). Instead, however, we have access to the gamboost algorithm, which uses boosting (as you learned about in chapter 8) to learn an ensemble of GAM models. Therefore, for this exercise, we’ll use the regr.gamboost learner. 
library(mboost)
gamTask <- makeRegrTask(data = ozoneForGam, target = "Ozone")

imputeMethod <- imputeLearner("regr.rpart")

gamImputeWrapper <- makeImputeWrapper("regr.gamboost",
                                      classes = list(numeric = imputeMethod))

gamFeatSelControl <- makeFeatSelControlSequential(method = "sfbs")

kfold <- makeResampleDesc("CV", iters = 10)

gamFeatSelWrapper <- makeFeatSelWrapper(learner = gamImputeWrapper,
                                        resampling = kfold,
                                        control = gamFeatSelControl)

####Next we cross-validate the model-building process

holdout <- makeResampleDesc("Holdout")

gamCV <- resample(gamFeatSelWrapper, gamTask, resampling = holdout)

gamCV

####Next we train the model

library(parallel)
library(parallelMap)

parallelStartSocket(cpus = detectCores())

gamModel <- train(gamFeatSelWrapper, gamTask)

parallelStop()

gamModelData <- getLearnerModel(gamModel, more.unwrap = TRUE)

####First, we train a boosted GAM using our gamTask. We can just use gamFeatSelWrapper as our learner, because this performs imputation and feature selection for us. To speed things along, we can parallelize the feature selection by running the parallel- StartSocket() function before running the train() function to actually train the model.
####We then extract the model information using the getLearnerModel() function. This time, because our learner is a wrapper function, we need to supply an additional argument, more.unwrap = TRUE, to tell mlr that it needs to go all the way down through the wrappers to extract the base model information.
####Now, let’s understand our model a little better by plotting the functions it learned for each of the predictor variables. This is as easy as calling plot() on our model information. We can also look at the residuals from the model by extracting them with the resid() function. This allows us to plot the predicted values (by extracting the $fitted() component) against their residuals to look for patterns that suggest a poor fit. We can also plot the quantiles of the residuals against the quantiles of a theoretical normal distribution, using qqnorm() and qqline(), to see if they are normally distributed.
####ecause we’re about to create a subplot for every predictor, and two for the residuals, we first divide the plotting device into nine parts using the mfrow argument of the par() function. We set this back again using the same function. You may have a different number of predictors than I do, as returned from your feature selection.

par(mfrow = c(3, 3))

plot(gamModelData, type = "l")

plot(gamModelData$fitted(), resid(gamModelData))

qqnorm(resid(gamModelData))

qqline(resid(gamModelData))

par(mfrow = c(1, 1))

### The resulting plot,  For each predictor, we get a plot of its value against how much that predictor contributes to the ozone estimate across its values. Lines show the shape of the functions learned by the algorithm, and we can see that they are all nonlinear.
#### The “rug” of tick marks at the base of each plot indicates the position of training cases. This helps us identify regions of each variable that have few cases, such as at the top end of the Visib variable. GAMs have the potential to overfit in regions with few cases.
#### Finally, looking at the residual plots, we can still see a pattern, which may indicate het- eroscedasticity in the data. The quantile plot shows that most of the residuals lie close to the diago- nal line, indicating that they approximate a normal distribution, with some deviation at the tails (which isn’t uncommon)

## Strengths and weaknesses of GAMs:

####While it often isn’t easy to tell which algorithms will perform well for a given task, here are some strengths and weaknesses that will help you decide whether GAMs will per- form well for you.
####The strengths of GAMs are as follows:
#### They produce models that are very interpretable, despite being nonlinear. 
#### They can handle both continuous and categorical predictors.
#### They can automatically learn nonlinear relationships in the data.

####The weaknesses of GAMs are these:
#### They still make strong assumptions about the data, such as homoscedasticity and the distribution of residuals (performance may suffer if these are violated).
#### GAMs have a propensity to overfit the training set.
#### GAMs can be particularly poor at predicting data outside the range of values of the training set.
#### They cannot handle missing data.









