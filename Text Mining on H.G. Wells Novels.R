# Text Mining on H.G. Wells Novels

library(gutenbergr)
library(dplyr)
library(tidytext)
library(ggplot2)
library(forcats)

## downloading Project Gutenberg the text of four books by HG Wells. We will combine these four books into a dataframe called 'books'.

titles <- c("The War of the Worlds",
            "The Time Machine",
            "Twenty Thousand Leagues under the Sea",
            "The Invisible Man: A Grotesque Romance")


books <- gutenberg_works(title %in% titles) %>%
  gutenberg_download(meta_fields = "title")

## Unnesting with n set to 2, we are examining pairs of two consecutive words, often called “bigrams”

wells_bigrams <- books %>%
  unnest_tokens(bigram, text, token = "ngrams", n = 2) %>%
  filter(!is.na(bigram))

head(wells_bigrams)

## Creating a sorted count of bigrams
wells_bigrams %>%
  count(bigram, sort = TRUE)

library(tidyr)

## seperating bigrams into indidvual words
bigrams_separated <- wells_bigrams %>%
  separate(bigram, c("word1", "word2"), sep = " ")

## filtering out bigrams with stop words occuring either in word1 or word2
bigrams_filtered <- bigrams_separated %>%
  filter(!word1 %in% stop_words$word) %>%
  filter(!word2 %in% stop_words$word)

# new bigram counts:
bigram_counts <- bigrams_filtered %>% 
  count(word1, word2, sort = TRUE)

head(bigram_counts)

## uniting bigrams after taking out stopwords
bigrams_united <- bigrams_filtered %>%
  unite(bigram, word1, word2, sep = " ")

## Plotting top 10 bigrams
bigram_plot <- bigrams_united %>%
  count(title, bigram, sort = TRUE) %>%
  arrange(desc(n))  %>%
  head(10)
bigram_plot

ggplot(bigram_plot, aes(reorder(bigram, n), n), color = title) +
  geom_col(show.legend = FALSE) +
  coord_flip()

# top 5 tf-idf plot

bigrams_united %>%
  count(title, bigram)  %>%
  bind_tf_idf(bigram, title, n) %>%
  arrange(desc(tf_idf)) %>%
  group_by(title) %>%
  slice(1:5)  %>%
  ungroup() %>%
  ggplot(aes(tf_idf, fct_reorder(bigram, tf_idf), fill = title))  +
  geom_col(show.legend = FALSE) +
  facet_wrap(~title, ncol = 2, scales = "free") +
  labs(x = "tf_idf", y = NULL)

## this plot shows the words in each of the 4 Wells novels with highest tf_idf

# Filtering out Words with "not" preceeding another word

bigrams_separated %>%
  filter(word1 == "not") %>%
  count(word1, word2, sort = TRUE)

## Word "not" preceeds another word for 463 instances.

## Continuing sentiment analysis:
## Using AFINN lexicon for numeric sentiment value for each word w/ positive or negative numbers for sentiments.

AFINN <- get_sentiments("afinn")

AFINN

## Examining the most frequent words that were preceded by "not and were associated with a sentiment

not_words <- bigrams_separated %>%
  filter(word1 == "not") %>%
  inner_join(AFINN, by = c(word2 = "word")) %>%
  count(word2, value, sort = TRUE)

not_words

not_words %>%
  mutate(contribution = n * value) %>%
  arrange(desc(abs(contribution))) %>%
  head(20) %>%
  mutate(word2 = reorder(word2, contribution)) %>%
  ggplot(aes(n * value, word2, fill = n * value > 0)) +
  geom_col(show.legend = FALSE) +
  labs(x = "Sentiment value * number of occurrences",
       y = "Words preceded by \"not\"")

## The plot shows  Words preceded by ‘not’ that had the greatest contribution to sentiment values, in either a positive or negative direction.

## Visualizing the notwork:

install.packages("igraph")

library(igraph)

head(bigram_counts)

### Filtering relatively common combination:

bigram_graph <- bigram_counts %>%
  filter(n > 20) %>%
  graph_from_data_frame()

bigram_graph

## loading ggraph package for developing graph from igraph data

install.packages("ggraph")
library(ggraph)

## Converting igraph object into ggraph and add layers

set.seed(2017)

ggraph(bigram_graph, layout = "fr") +
  geom_edge_link()  +
  geom_node_point() +
  geom_node_text(aes(label = name), vjust = 1, hjust = 1)

## This plot shows Common bigrams in Well’s novels, showing those that occurred more than 20 times and where neither word was a stop word

## Adding  polishing operations to make a better graph:

set.seed(2020)

a <- grid::arrow(type = "closed", length = unit(.15, "inches"))

ggraph(bigram_graph, layout = "fr") +
  geom_edge_link(aes(edge_alpha = n), show.legend = FALSE,
                 arrow = a, end_cap = circle(.07, 'inches')) +
  geom_node_point(color = "lightblue", size = 5) +
  geom_node_text(aes(label = name), vjust = 1, hjust = 1) +
  theme_void()

## Made the links darker the more common the bigram is. Used arrows at the end of the line toward the second word. Colorized the central node.

## Making a function to count bigrams:
### This code unnests tokens with n = 2 as bigrams, seperate the bigrams to take out stop words from each word and then create a count.

count_bigrams <- function(dataset) {
  dataset %>%
    unnest_tokens(bigram, text, token = "ngrams", n = 2) %>%
    separate(bigram, c("word1", "word2"), sep = " ") %>%
    filter(!word1 %in% stop_words$word,
           !word2 %in% stop_words$word) %>%
    count(word1, word2, sort = TRUE)
}

## Making a function for visualization:
## This code creates a visualization arrows, layout type, links, node points, colors, and text labels.

visualize_bigrams <- function(bigrams) {
  set.seed(2016)
  a <- grid::arrow(type = "closed", length = unit(.15, "inches"))
  
  bigrams %>%
    graph_from_data_frame() %>%
    ggraph(layout = "fr") +
    geom_edge_link(aes(edge_alpha = n), show.legend = FALSE, arrow = a) +
    geom_node_point(color = "lightblue", size = 5) +
    geom_node_text(aes(label = name), vjust = 1, hjust = 1) +
    theme_void()
}

