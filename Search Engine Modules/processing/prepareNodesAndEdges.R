#prepare nodes and edges
library(dplyr)
library(tidyverse)
library(tidytext)
library(geoneta)


# prepare nodes
Article_text_withcategory  <- read.csv("~/Documents/shiny/search2/Article_text_withcategory.csv")
Article_text_withcategory<-Article_text_withcategory[-c(1)]

nodes<-Article_text_withcategory[c(1,5)]
nodes<-distinct(nodes, ID, .keep_all = TRUE)
nodes$label <-nodes$ID
nodes<-nodes[c(1,3,2)]


names(nodes)[names(nodes) == "ID"] <- "id"
names(nodes)[names(nodes) == "Attack"] <- "group"

nodes<-nodes[sample(nrow(nodes), 200), ]
#prepare edges

Article_text_withcategory  <- read.csv("~/Documents/shiny/search2/Article_text_withcategory.csv")
Article_text_withcategory<-Article_text_withcategory[-c(1)]
Article_text_withcategory<-Article_text_withcategory[Article_text_withcategory$argument != 'O',]
Article_text_withcategory<-Article_text_withcategory[c(1,2)]
Article_text_withcategory <- Article_text_withcategory %>%
  anti_join(stop_words, by= c("word" = "word"))


word<-data.frame(table(Article_text_withcategory$word))
word<-word[word$Freq >=2,]
word<-word[word$Freq <675,]
word<-word[word$Var1 !='-' ,]
word<-word[word$Var1 !=']' ,]
word<-word[word$Var1 !='â?????Ts' ,]
word<-word[word$Var1 !=',' ,]
word<-word[word$Var1 !="'s" ,]
word<-word[word$Var1 !="(" ,]
word <- word %>%
  filter(!str_detect(Var1, "[:punct:]|[:digit:]"))
word<-word[-c(1:6),]


Article_text_withcategory <- Article_text_withcategory[which(Article_text_withcategory$word%in%word$Var1),]

#unique sentence number
unique_sentence<-unique(Article_text_withcategory$ID)
Article_text_withcategory<-Article_text_withcategory[order(Article_text_withcategory$word)]
Article_text_withcategory <- Article_text_withcategory[order(Article_text_withcategory$word),]


#sample 200 
Article_text_withcategory_sample<- Article_text_withcategory[which(Article_text_withcategory$ID%in%nodes$id),]



a <- Article_text_withcategory_sample
a<-a[c(2,1)]
names(a)[names(a) == "word"] <- "ID1"
names(a)[names(a) == "ID"] <- "Code"
names(a)[names(a) == "ID1"] <- "ID"
c<-data.frame(a %>% full_join(a, by="ID") %>% group_by(Code.x,Code.y) %>% summarise(length(unique(ID))) %>% filter(Code.x!=Code.y))

edges<-c

names(edges)[names(edges) == "Code.x"] <- "from"
names(edges)[names(edges) == "Code.y"] <- "to"
names(edges)[names(edges) == "length.unique.ID.."] <- "width"


save(nodes, file = '~/Documents/shiny/search3/nodes1.RData')
save(edges, file = '~/Documents/shiny/search3/edges1.RData' )


#df <- data.frame(ID=c(1,1,1,2,2,2,3,3,3,3,4,4), 
#                 Code=c("A", "B", "C", "B", "C", "D", "C", "A", "D", "B", "D", "B"), stringsAsFactors =FALSE)
#b<-data.frame(df %>% full_join(df, by="ID") %>% group_by(Code.x,Code.y) %>% summarise(length(unique(ID))) %>% filter(Code.x!=Code.y))

#Article_text_withcategory %>% full_join(Article_text_withcategory, by="word") %>% group_by(ID.x,ID.y) %>% summarise(length(unique(word))) %>% filter(ID.x!=ID.x)



