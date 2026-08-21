library(shiny)
library(dplyr)
library(ggplot2)
library(kohonen)
library(randomForest)
library(e1071)
library(clusterCrit)
library(umap)
library(ps)

options(shiny.maxRequestSize = 20 * 1024^2)

FEATURES <- c(
  'DEP_DELAY',
  'CRS_ELAPSED_TIME',
  'DISTANCE',
  'DAY_OF_WEEK',
  'CARRIER_DELAY',
  'WEATHER_DELAY',
  'LATE_AIRCRAFT_DELAY',
  'NAS_DELAY',
  'OP_UNIQUE_CARRIER',
  'ORIGIN'
)

preprocess_data <- function(path, max_rows = 200000) {
  
  df <- read.csv(path)
  
  if(nrow(df) > max_rows) {
    set.seed(42)
    df <- df[sample(nrow(df), max_rows), ]
    print(paste("Sampled", max_rows, "rows"))
  }
  
  names(df) <- trimws(names(df))
  df <- df[, FEATURES]
  df <- na.omit(df)
  
  print("Before conversion:")
  print(sapply(df, class))
  
  df$OP_UNIQUE_CARRIER <- as.numeric(as.factor(df$OP_UNIQUE_CARRIER))
  df$ORIGIN <- as.numeric(as.factor(df$ORIGIN))
  
  print("After conversion:")
  print(sapply(df, class))
  
  df_numeric <- as.data.frame(lapply(df, as.numeric))
  
  na_count <- sum(is.na(df_numeric))
  if(na_count > 0) {
    print(paste("WARNING:", na_count, "NA values found after conversion"))
    df_numeric <- na.omit(df_numeric)
  }
  
  X <- scale(df_numeric)
  
  print("Scaling completed successfully")
  
  return(list(
    X = X,
    df = df
  ))
}


describe_cluster <- function(row) {
  
  dep <- as.numeric(row['DEP_DELAY']) 
  
  weather <- as.numeric(row['WEATHER_DELAY'])
  
  carrier <- as.numeric(row['CARRIER_DELAY'])
  
  nas <- as.numeric(row['NAS_DELAY'])
  
  late <- as.numeric(row['LATE_AIRCRAFT_DELAY'])
  
  distance <- as.numeric(row['DISTANCE'])
  
  
  if(weather > 30) {
    
    return('Задержки из-за погодных условий')
    
  } else if(carrier > 30) {
    
    return('Задержки по вине авиакомпании')
    
  } else if(nas > 30) {
    
    return('Задержки из-за загруженности воздушного пространства')
    
  } else if(late > 30) {
    
    return('Задержки из-за позднего прибытия самолёта')
    
  } else if(distance > 2000 &&
            dep > 20) {
    
    return('Задержки на дальнемагистральных рейсах')
    
  } else if(dep > 60) {
    
    return('Серьёзные системные задержки')
    
  } else if(dep < 10) {
    
    return('Рейсы без задержек')
    
  } else {
    
    return('Смешанные операционные задержки')
  }
}


som_cluster <- function(X) {
  
  grid <- somgrid(
    xdim = 3,
    ydim = 3,
    topo = "rectangular"
  )
  
  model <- som(
    X,
    grid = grid,
    rlen = 1000,
    alpha = c(0.05, 0.01)
  )
  
  labels <- model$unit.classif
  
  return(labels)
}


rf_cluster <- function(X) {
  
  set.seed(42)
  
  y <- sample(c(0, 1), nrow(X), replace = TRUE)
  
  rf <- randomForest(
    x = X,
    y = as.factor(y),
    ntree = 100,
    maxnodes = 50,
    nodesize = 10,
    replace = TRUE
  )
  
  leaves <- predict(rf, X, nodes = TRUE)
  
  print(paste("Структура leaves:", class(leaves)))
  print(paste("Размерность:", paste(dim(leaves), collapse = " x ")))
  
  
  if(is.null(dim(leaves))) {
    leaves <- matrix(leaves, ncol = 1)
    print("leaves преобразован в матрицу")
  }
  
  unique_rows <- nrow(unique(leaves))
  print(paste("Уникальных точек данных:", unique_rows))
  
  if(is.null(unique_rows) || length(unique_rows) == 0) {
    warning("Не удалось определить количество уникальных точек")
    return(rep(1, nrow(X)))
  }
  
  if(unique_rows < 2) {
    warning("Все точки одинаковые. Возвращаем 1 кластер.")
    return(rep(1, nrow(X)))
  }
  
  n_clusters <- min(3, unique_rows)
  print(paste("Будет создано кластеров:", n_clusters))
  
  set.seed(42)
  
  result <- tryCatch({
    km <- kmeans(leaves, centers = n_clusters, nstart = 10)
    return(km$cluster)
  }, error = function(e) {
    warning("Ошибка kmeans: ", e$message)
    
    if(nrow(leaves) > 1 && unique_rows >= 2) {
      d <- dist(leaves)
      hc <- hclust(d, method = "ward.D2")
      return(cutree(hc, k = n_clusters))
    } else {
      return(rep(1, nrow(X)))
    }
  })
  
  return(result)

}


ocsvm_cluster <- function(X) {
  
  print("Я здесь")
  
  gamma_auto <- 1 / ncol(X)
  
  model <- svm(
    X,
    type = "one-classification",
    kernel = "radial",         
    gamma = gamma_auto,        
    nu = 0.15,                 
    tolerance = 0.001,
    cache = 500     
  )
  
  pred <- predict(
    model,
    X
  )
  
  labels <- ifelse(
    pred,
    1,
    -1
  )
  print("Теперь тут")
  return(labels)
}


calculate_metrics <- function(X, labels) {
  
  if(length(unique(labels)) < 2) {
    
    return(data.frame(
      Silhouette = NA,
      Davies_Bouldin = NA,
      Calinski_Harabasz = NA
    ))
  }
  
  metrics <- intCriteria(
    as.matrix(X),
    as.integer(labels),
    c(
      "Silhouette",
      "Davies_Bouldin",
      "Calinski_Harabasz"
    )
  )
  
  return(data.frame(
    
    Silhouette =
      metrics$silhouette,
    
    Davies_Bouldin =
      metrics$davies_bouldin,
    
    Calinski_Harabasz =
      metrics$calinski_harabasz
  ))
}

save_plot <- function(
    X,
    labels,
    algo,
    summary
) {
  
  umap_result <- umap(X)
  
  embedding <- as.data.frame(umap_result$layout)
  colnames(embedding) <- c("UMAP1", "UMAP2")
  
  cluster_labels <- labels
  cluster_full_names <- paste0("Кластер ", summary$Cluster, ": ", summary$Description)
  
  names(cluster_full_names) <- summary$Cluster
  
  embedding$Cluster <- factor(
    labels,
    levels = summary$Cluster,
    labels = cluster_full_names[as.character(summary$Cluster)]
  )
  
  centers <- embedding %>%
    group_by(Cluster) %>%
    summarise(
      UMAP1 = mean(UMAP1),
      UMAP2 = mean(UMAP2)
    )
  
  p <- ggplot(embedding, aes(UMAP1, UMAP2, color = Cluster)) +
    geom_point(alpha = 0.8, size = 2) +
    
    geom_label(
      data = centers,
      aes(UMAP1, UMAP2, label = gsub(":.*$", "", Cluster)),
      fontface = 'bold',
      size = 4,
      fill = "white",
      alpha = 0.85,
      color = "black",
      label.size = 0.8,
      label.padding = unit(0.3, "lines"),
      show.legend = FALSE
    ) +
    
    theme_minimal() +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      legend.position = "right",
      legend.title = element_text(face = "bold", size = 10),
      legend.text = element_text(size = 9)
    ) +
    ggtitle(paste(algo, "- UMAP Clustering")) +
    labs(color = "Кластеры")
  
  filename <- paste0(algo, "_UMAP.png")
  
  ggsave(
    filename,
    plot = p,
    width = 16,
    height = 8,
    dpi = 300
  )
  
  return(filename)
}

ui <- fluidPage(
  
  titlePanel(
    "Flight Delay Clustering"
  ),
  
  sidebarLayout(
    
    sidebarPanel(
      
      fileInput(
        "file",
        "Upload CSV"
      ),
      
      selectInput(
        "algo",
        "Choose Algorithm",
        choices = c(
          "SOM",
          "RandomForest",
          "OneClassSVM"
        )
      ),
      
      actionButton(
        "run",
        "Run Clustering"
      )
    ),
    
    mainPanel(
      
      verbatimTextOutput(
        "results"
      )
    )
  )
)


server <- function(input, output) {
  
  observeEvent(input$run, {
    
    req(input$file)
    
    data <- preprocess_data(
      input$file$datapath
    )
    
    X <- data$X
    
    df <- data$df
    
    gc()
    
    memory_before <- ps_memory_info(
      ps_handle()
    )["rss"] / (1024^2)
    
    start_time <- Sys.time()
    
    labels <- switch(
      
      input$algo,
      
      "SOM" =
        som_cluster(X),
      
      "RandomForest" =
        rf_cluster(X),
      
      "OneClassSVM" =
        ocsvm_cluster(X)
    )
    
    end_time <- Sys.time()
    
    gc()
    
    memory_after <- ps_memory_info(
      ps_handle()
    )["rss"] / (1024^2)
    
    execution_time <- round(
      as.numeric(
        difftime(
          end_time,
          start_time,
          units = "secs"
        )
      ),
      4
    )
    
    memory_used <- round(
      memory_after - memory_before,
      4
    )
    
    if (memory_used <= 0) {
      memory_used <- 0.01
    }
    
    metrics <- calculate_metrics(
      X,
      labels
    )
    
    df$Cluster <- labels
    
    summary <- df %>%
      group_by(Cluster) %>%
      summarise(
        DEP_DELAY = mean(DEP_DELAY),          
        CARRIER_DELAY = mean(CARRIER_DELAY),  
        WEATHER_DELAY = mean(WEATHER_DELAY),   
        NAS_DELAY = mean(NAS_DELAY),           
        LATE_AIRCRAFT_DELAY = mean(LATE_AIRCRAFT_DELAY),
        DISTANCE = mean(DISTANCE)              
      )
    
    summary$Description <- apply(
      summary,
      1,
      describe_cluster
    )
    
    image_file <- save_plot(
      X,
      labels,
      input$algo,
      summary
    )
    
    output$results <- renderPrint({
      
      cat(
        "Algorithm:",
        input$algo,
        "\n\n"
      )
      
      cat(
        "Execution Time:",
        execution_time,
        "sec\n"
      )
      
      cat(
        "Memory Used:",
        memory_used,
        "MB\n\n"
      )
      
      cat("Metrics:\n\n")
      
      print(metrics)
      
      cat(
        "\nCluster Descriptions:\n\n"
      )
      
      print(
        summary[, c(
          'Cluster',
          'Description'
        )]
      )
      
      cat(
        "\nPlot saved as:\n"
      )
      
      cat(
        image_file,
        "\n"
      )
    })
  })
}

shinyApp(
  ui = ui,
  server = server
)