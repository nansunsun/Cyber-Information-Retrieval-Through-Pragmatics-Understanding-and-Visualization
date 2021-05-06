##give category
Article_text <- read_csv("Article_text.csv")
Article_text$Attack = Article_text$nugget
Article_text$Attack[Article_text$Attack=="O"] <- NA
Article_text<-Article_text %>% fill(Attack,.direction = c("down"))
Article_text$Attack<-substring(Article_text$Attack, 3)
Article_text[1, 5] = 'DiscoverVulnerability'
Article_text[2, 5] = 'DiscoverVulnerability'
write.csv(Article_text, "C:/Users/mzz/Downloads/Article_text_withcategory.csv")


Tweets_text <- read_csv("C:/Users/mzz/Downloads/Tweets_text.csv")
Tweets_text$Attack = Tweets_text$nugget
Tweets_text$Attack[Tweets_text$Attack=="O"] <- NA
Tweets_text<-Tweets_text %>% fill(Attack,.direction = c("down"))
Tweets_text$Attack<-substring(Tweets_text$Attack, 3)

write.csv(Tweets_text, "C:/Users/mzz/Downloads/Tweets_text_withcategory.csv")


text <- read_csv("C:/Users/mzz/Downloads/text.csv")
text$Attack = text$nugget
text$Attack[text$Attack=="O"] <- NA
text<-text %>% fill(Attack,.direction = c("down"))
text$Attack<-substring(text$Attack, 3)

write.csv(text, "C:/Users/mzz/Downloads/text_withcategory.csv")
