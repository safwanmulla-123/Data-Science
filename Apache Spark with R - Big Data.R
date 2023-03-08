# Apache Spark with R - Big Data

library(sparklyr)
library(dplyr)
library(corrr)
library(ggplot2)

## Establishing a connection with my local spark network
sc <- spark_connect(master = "local")

## Copying the mtcars dataset into spark as cars
cars <- copy_to(sc, mtcars)

## Below we group the data in spark by cyl and summarize the sum of mpg.
## After that we collect that data and pull and print in R

car_group <- cars %>%
  group_by(cyl) %>%
  summarize(mpg = sum(mpg, na.rm=TRUE)) %>%
  collect() %>%
  print()

## Below we group the data in spark by cyl and summarize the smean/average of mpg.
## After that we collect that data and pull and print in R

car_group1 <- cars %>%
  group_by(cyl) %>%
  summarize(mpg = mean(mpg, na.rm=TRUE)) %>%
  collect() %>%
  print()

ggplot(aes(as.factor(cyl), mpg), data = car_group) + 
  geom_col(fill = "#999999") + coord_flip()
## Here we plot a vertical bar plot of cyl vs mpg sum with grey fill.
## It shows the sum of mpgs for the 3 cyls - 4,6,8.

ggplot(aes(as.factor(cyl), mpg), data = car_group1) + 
  geom_col(fill = "#999999") + coord_flip()
## Here we plot a vertical bar plot of cyl vs avg/mean mpg data with grey fill.
## This plot is a more helpful that our last plot of sum of mpgs.
## From this plt we can conclude that the 4, 6, 8 cyl cars have an avg mpg of approx. 28, 20, 15 respectively.

ggplot(aes(as.factor(cyl), mpg), data = car_group) +
  geom_col(fill = "red")
## Here we plot a vertical bar plot of cyl vs mpg sum with red fill.
## It shows the sum of mpgs for the 3 cyls - 4,6,8.


ggplot(aes(as.factor(cyl), mpg), data = car_group1) +
  geom_col(fill = "red")
## Here we plot a vertical bar plot of cyl vs avg/mean mpg data with red fill.
## This plot is a more helpful that our last plot of sum of mpgs.
## From this plt we can conclude that the 4, 6, 8 cyl cars have an avg mpg of approx. 28, 20, 15 respectively.

spark_disconnect(sc)
