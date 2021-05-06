library(visNetwork)
library(geoneta)
library(igraph)

load(file = "~/Documents/shiny/search3/lesmis.rda")
nodes <- as.data.frame(lesmis[2])
colnames(nodes) <- c("id", "label")
nodes$id <- nodes$label
edges <- as.data.frame(lesmis[1])
colnames(edges) <- c("from", "to", "width")
#Create graph for Louvain
graph <- graph_from_data_frame(edges, directed = FALSE)

#Louvain Comunity Detection
cluster <- cluster_louvain(graph)

cluster_df <- data.frame(as.list(membership(cluster)))
cluster_df <- as.data.frame(t(cluster_df))
cluster_df$label <- rownames(cluster_df)

#Create group column
nodes <- left_join(nodes, cluster_df, by = "label")
colnames(nodes)[3] <- "group"
visNetwork(nodes, edges)


visNetwork(nodes, edges, width = "100%") %>%
  visIgraphLayout() %>%
  visNodes(
    shape = "dot",
    color = list(
      background = "#0085AF",
      border = "#013848",
      highlight = "#FF8000"
    ),
    shadow = list(enabled = TRUE, size = 10)
  ) %>%
  visEdges(
    shadow = FALSE,
    color = list(color = "#0085AF", highlight = "#C62F4B")
  ) %>%
  visOptions(highlightNearest = list(enabled = T, degree = 1, hover = T),
             selectedBy = "group") %>% 
  visLayout(randomSeed = 11)

save(nodes, file = "C:/Users/mzz/Downloads/nodes.RData")
save(edges, file = "C:/Users/mzz/Downloads/edges.RData")
load("nodes")
