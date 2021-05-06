library(shiny)
library(visNetwork)
library(igraph)

server<-shinyServer(function(input, output) {
    
  output$network <- renderVisNetwork({
    a<-get(load("nodes.RData"))
    b<-get(load("edges1.RData"))
    
    legendNodes <- data.frame(
      label = c("Data Breach","Phishing","Ransom","Discover Vulnerability","Patch Vulnerability"),
      color.background = c("red","skyblue","yellow","limegreen","orchid"),
      color.border = c("firebrick","blue","gold","green","purple"),
      shape =c("dot","dot","dot","dot","dot")
    )
    
   
    
    visNetwork(a, b) %>%
      visIgraphLayout()%>%
      visOptions(highlightNearest = TRUE, manipulation = TRUE,
                 selectedBy = "group", nodesIdSelection = TRUE)%>%
      
      visLegend(useGroups = FALSE,addNodes = legendNodes, position ="right"
      )}) 
    
    }) 
  

ui<-shinyUI(
  fluidPage(
    titlePanel("Security Event Visualisation"),
    visNetworkOutput("network")
      )
)

shinyApp(ui = ui, server = server)