library(data.table)
TwitterTraffic <- read.csv("~/Documents/GitHub/CASIE/data/finaldataset/TwitterTraffic.csv")
TwitterTraffic<-TwitterTraffic[c(2,3,5,7,4,11,3,10)]
names(TwitterTraffic)[1]<-paste("ID")
names(TwitterTraffic)[2]<-paste("title")
names(TwitterTraffic)[3]<-paste("point")
names(TwitterTraffic)[4]<-paste("comment")
names(TwitterTraffic)[5]<-paste("haha")
names(TwitterTraffic)[6]<-paste("author")
names(TwitterTraffic)[7]<-paste("content")
TwitterTraffic$haha<-as.Date(TwitterTraffic$haha)



TwitterTraffic$haha<-str_replace_all(TwitterTraffic$haha, "[[:punct:]]", "")
TwitterTraffic_phishing <- TwitterTraffic[grep("vulnera", TwitterTraffic$title), ]
TwitterTraffic_phishing <- na.omit(TwitterTraffic_phishing)
write.csv(TwitterTraffic_phishing,"~/Documents/Reddit_shiny-master/temp/Vulnerability.csv", row.names = FALSE)
TwitterTraffic_databreach<- TwitterTraffic[grep("breach", TwitterTraffic$title), ]
write.csv(TwitterTraffic_databreach,"~/Documents/Reddit_shiny-master/data/TwitterTraffic_breach.csv", row.names = FALSE)
