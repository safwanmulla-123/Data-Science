library(ggplot2)
library(tidyverse)
library(socviz)

by_country <- organdata %>% 
  group_by(consent_law, country) %>% 
  summarize(donors_mean= mean(donors, na.rm = TRUE),
            donors_sd = sd(donors, na.rm = TRUE), 
            gdp_mean = mean(gdp, na.rm = TRUE), 
            health_mean = mean(health, na.rm = TRUE), 
            roads_mean = mean(roads, na.rm = TRUE), 
            cerebvas_mean = mean(cerebvas, na.rm = TRUE))
by_country

#produce a scatterplot of the by_country data with the points colored by consent_law

ggplot(data = by_country,
       mapping = aes(x = donors_mean, y = reorder(country, donors_mean),
                     color = consent_law)) + 
  geom_point()

#Using facet_wrap() split the consent_law variable into two panels and rank the countries by donation rate within the panels

ggplot(data = by_country,
       mapping = aes(x = donors_mean,
                     y = reorder(country, donors_mean))) + 
  geom_point(size=3) +
  facet_wrap(~ consent_law, scales = "free_y", ncol = 1)
#Shows twso differtn grids for informed an presumed values

#Use geom_pointrange() to create a dot and whisker plot showing the mean of donors and a confidence interval. 

ggplot(data = by_country, mapping = aes(x = reorder(country, donors_mean), 
                                        y = donors_mean)) +
  geom_pointrange(mapping = aes(ymin = donors_mean - donors_sd, ymax = donors_mean + donors_sd))  +
  coord_flip()
#Shows the range within which the donor values, which has a point to represent the mean

#Create a scatterplot of roads_mean v. donors_mean with the labels identifying the country sitting to the right or left of the point

ggplot(data = by_country, mapping = aes(x = roads_mean, y = donors_mean)) +
  geom_point() + 
  geom_text(mapping = aes(label = country)) +
  coord_flip()

#load the ggrepel() library
library(ggrepel)

#using the elections_historic data, plot the presidents popular vote percentage v electoral college vote percentage. draw axes at 50% for each attribute and use geom_text_repel() to keep the labels from obscuring the points. 
colnames(elections_historic)
ggplot(elections_historic, aes(x = popular_pct, y = ec_pct, label = winner_label)) +
  geom_hline(yintercept = 0.5, size = 1.4, color = "gray80") + 
  geom_vline(xintercept = 0.5, size = 1.4, color = "gray80") + 
  geom_point() +
  geom_text_repel()

#What is the electoral college?
?elections_historic
#ec_votes. Electoral college votes cast for winner.

#create a new binary value column in organdata called 'ind' populated by determining whether the ccode is "Spa" or "Ita" and the year is after than 1998.

organdata$ind <- organdata$ccode %in% c("Ita", "Spa") & organdata$year > 1998

#Shows a graphs classifying  Spa or Ita.

#create an organdata plot of Roads v. Donors and map the ind attribute to the color aesthetic. Label those points with the ccode and suppress the legends.

ggplot(data = organdata, mapping = aes(x = roads,
                                       y = donors, color = ind)) +
  geom_point() +
  geom_text_repel(data = subset(organdata, ind),
                  mapping = aes(label = ccode)) + 
  guides(label = FALSE, color = FALSE)


#Add a label in a rectangle to the previous plot that says "Spa = Spain & Ita = Italy".

ggplot(data = organdata, mapping = aes(x = roads,
                                       y = donors, color = ind)) +
  geom_point() +
  geom_text_repel(data = subset(organdata, ind),
                  mapping = aes(label = ccode)) + 
  guides(label = FALSE, color = FALSE) + 
  annotate(geom = "rect", xmin = 50, xmax = 100,
           ymin = 30, ymax = 35, fill = "red", alpha = 0.2) +
  annotate(geom = "text", x = 61, y = 33, label = "Spa = Spain & Ita = Italy", hjust = 0)

















