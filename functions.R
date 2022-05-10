train_test_split <- function(dataset, ratio, cols) {
    split <- sample(c(rep(0, ratio * nrow(dataset)), rep(1, (1-ratio) * nrow(dataset))))
    
    train <- dataset[split == 0,]
    test <- dataset[split == 1,]
    
    train <- train[,cols]
    test <- test[,cols]
    
    return(list(train = train,test = test))
}

classification_metrics <- function(true_labels, pred_labels) {
    cm <- table(true_labels,pred_labels)
    n <- sum(cm)
    nc = nrow(cm)
    correct = diag(cm)
    rowsums = apply(cm, 1, sum)
    colsums = apply(cm, 2, sum)
    p = rowsums / n
    accuracy = sum(correct) / n
    precision = correct / colsums 
    recall = correct / rowsums 
    f1 = 2 * precision * recall / (precision + recall) 
    micro <- data.frame(precision, recall, f1) 
    macroPrecision = mean(precision)
    macroRecall = mean(recall)
    macroF1 = mean(f1)
    macro <- data.frame(macroPrecision, macroRecall, macroF1)
    print("Confusion Matrix:")
    print(cm)
    print("Accuracy:")
    print(accuracy)
    print("Per class metrics: ")
    print(micro)
    print("Macro metrics: ")
    print(macro)
    return(list(Confusion_Matrix = cm,Accuracy = accuracy, Micro = micro, Macro = macro))
}



ml_pipeline <- function(data_train_test, algorithm, split = "gini", pretty = 1, bootstraps = 500, bagging = FALSE, interaction_depth = 1) {
    if (algorithm == "decision_trees") {
        dataset.dt <- tree(as.factor(Order) ~.,
                           data = data_train_test[[1]],
                           split = split)
        plot(dataset.dt)
        text(dataset.dt, pretty = pretty)
        dataset.predicted <- predict(dataset.dt, data_train_test[[2]], type = "class")
        print(summary(dataset.dt))
        metrics <- classification_metrics(data_train_test[[2]]$Order, dataset.predicted)
        return(dataset.dt)
    }
    if (algorithm == "random_forest") {
        if (bagging == FALSE) {
            dataset.rf <- randomForest(as.factor(Order) ~ .,
                                       data = data_train_test[[1]],
                                       importance = TRUE,
                                       proximity = TRUE,
                                       ntrees = bootstraps)
        } else {
            dataset.rf <- randomForest(as.factor(Order) ~ .,
                                       data = data_train_test[[1]],
                                       importance = TRUE,
                                       proximity = TRUE,
                                       mtry = ncol(data_train_test[[1]])-1,
                                       ntrees = bootstraps)
        }
        plot(dataset.rf)
        dataset.predicted <- predict(dataset.rf, data_train_test[[2]], type = "class")
        print(summary(dataset.rf))
        metrics <- classification_metrics(data_train_test[[2]]$Order, dataset.predicted)
        print("Variable Importance:")
        print(importance(dataset.rf))
        varImpPlot(dataset.rf)
        return(dataset.rf)
    }
    if (algorithm == "logistic") {
        dataset.lr <- multinom(as.factor(Order) ~., data = data_train_test[[1]])
        dataset.lr.predicted <- predict(dataset.lr, data_train_test[[2]], type = "class")
        print(summary(dataset.lr))
        metrics <- classification_metrics(data_train_test[[2]]$Order, dataset.lr.predicted)
        return(dataset.lr)
    }
    if (algorithm == "boosting") {
        dataset.boost <- gbm(as.factor(Order) ~., data = data_train_test[[1]], distribution = "multinomial",
                             n.trees = bootstraps, interaction.depth = interaction_depth)
        print(summary(dataset.boost))
        dataset.boost.predicted <- predict(dataset.boost, data_train_test[[2]])
        metrics <- classification_metrics(data_train_test[[2]]$Order, dataset.boost.predicted)
        return(dataset.boost)
    }
    
}
