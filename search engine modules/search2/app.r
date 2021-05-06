# Load the ggplot2 package which provides
# the 'mpg' dataset.
library(readr)
library(ggplot2)
library(shiny)
text <- read_csv("Article_text_withcategory.csv")
text <- text[-c(1)]
ui<- fluidPage(
  titlePanel("Cybersecurity Event Nugget and Argument"),
  
  # Create a new Row in the UI for selectInputs
  fluidRow(
    column(3,
           selectInput("ID",
                       "ID:",
                       c("All",
                         unique(as.character(text$ID))))
    ),
    column(3,
           selectInput("Attack",
                       "Attack category:",
                       c("All",
                         unique(as.character(text$Attack))))
    ),
    column(3,
           selectInput("nugget",
                       "Event nugget:",
                       c("All",
                         unique(as.character(text$nugget))))
    ),
    column(3,
           selectInput("argument",
                       "Argument:",
                       c("All",
                         unique(as.character(text$argument))))
    )
  ),
  # Create a new row for the table.
  DT::dataTableOutput("table")
)


# Load the ggplot2 package which provides
# the 'mpg' dataset.
library(ggplot2)

server<-
function(input, output) {
  
  # Filter data based on selections
  output$table <- DT::renderDataTable(DT::datatable({
    data <- text
    if (input$nugget != "All") {
      data <- data[data$nugget == input$nugget,]
    }
    if (input$argument != "All") {
      data <- data[data$argument == input$argument,]
    }
    if (input$Attack != "All") {
      data <- data[data$Attack == input$Attack,]
    }
    if (input$ID != "All") {
      data <- data[data$ID == input$ID,]
    }
    data
  }))
  
}

shinyApp(ui = ui, server = server)

