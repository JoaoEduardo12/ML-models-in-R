
##### Data/Library preparation #####

# necessary libraries
library(tree)
library(randomForest)
library(nnet)
library(MASS)

# necessary functions
source("functions.R")

# read csv data
dataset <- read.csv("PCG_concatenated.csv")

# define specific columns
nuc <- c(2,3:11)
rscu <- c(2,12:71)
rscu_pcg <- c(8,9:68)
rscu_long <- c(2,7:66)

# set random seed
set.seed(42)
# prepare train and test data sets
data_train_test <- train_test_split(dataset, 0.7, rscu)

##### Decision Trees #####

# run our ml pipeline for decision trees
pcg.dt <- ml_pipeline(data_train_test,"decision_trees", split = "deviance", pretty = 1)
pcg.dt

##### Cost Complexity Pruning for decision trees #####

# decision tree cost complexity pruning
pcg.cv.dt <- cv.tree(pcg.dt, FUN = prune.misclass)
pcg.cv.dt

#'Next, we consider whether pruning the tree might lead to improved
# results. The function cv.tree() performs cross-validation in order to
# determine the optimal level of tree complexity; cost complexity pruning
# is used in order to select a sequence of trees for consideration. We use
# the argument FUN=prune.misclass in order to indicate that we want the
# classiﬁcation error rate to guide the cross-validation and pruning process,
# rather than the default for the cv.tree() function, which is deviance. The
# cv.tree() function reports the number of terminal nodes of each tree con-
#     sidered ( size ) as well as the corresponding error rate and the value of the
# cost-complexity parameter used

# dev corresponds to the cross-validation error
# rate in this instance. The tree with 10 terminal nodes results in the lowest
# cross-validation error rate, with 58 cross-validation errors. We plot the error
# rate as a function of both size and k .

par(mfrow=c(1,2))
plot(pcg.cv.dt$size, pcg.cv.dt$dev, type = "b")
plot(pcg.cv.dt$k,pcg.cv.dt$dev,type ="b")

# We now apply the prune.misclass() function in order to prune the tree to
# obtain the nine-node tree.

pcg.dt.pruned = prune.misclass(pcg.dt, best = 11)
dev.off()
plot(pcg.dt.pruned)
text(pcg.dt.pruned, pretty =0)

pred_values.dt.pruned <- predict(pcg.dt.pruned, data_train_test[[2]], type = "class")
summary(pcg.dt.pruned)
metrics <- classification_metrics(data_train_test[[2]]$Order,pred_values.dt.pruned)

##### Fitting DT to each individual gene #####
atp6 <- read.csv("ATP6.csv")
data_train_test_atp6 <- train_test_split(atp6, 0.7, rscu_pcg)
atp6.dt <- ml_pipeline(data_train_test_atp6,"decision_trees", split = "deviance", pretty = 1)

atp8 <- read.csv("ATP8.csv")
data_train_test_atp8 <- train_test_split(atp8, 0.7, rscu_pcg)
atp8.dt <- ml_pipeline(data_train_test_atp8,"decision_trees",split = "deviance", pretty = 1)

cox1 <- read.csv("COX1.csv")
data_train_test_cox1 <- train_test_split(cox1, 0.7, rscu_pcg)
cox1.dt <- ml_pipeline(data_train_test_cox1,"decision_trees",split = "deviance", pretty = 1)

cox2 <- read.csv("COX2.csv")
data_train_test_cox2 <- train_test_split(cox2, 0.7, rscu_pcg)
cox2.dt <- ml_pipeline(data_train_test_cox2,"decision_trees",split = "deviance", pretty = 1)

cox3 <- read.csv("COX3.csv")
data_train_test_cox3 <- train_test_split(cox3, 0.7, rscu_pcg)
cox3.dt <- ml_pipeline(data_train_test_cox3,"decision_trees",split = "deviance", pretty = 1)

cob <- read.csv("COB.csv")
data_train_test_cob <- train_test_split(cob, 0.7, rscu_pcg)
cob.dt <- ml_pipeline(data_train_test_cob,"decision_trees",split = "deviance", pretty = 1)

nad1 <- read.csv("NAD1.csv")
data_train_test_nad1 <- train_test_split(nad1, 0.7, rscu_pcg)
nad1.dt <- ml_pipeline(data_train_test_nad1,"decision_trees",split = "deviance", pretty = 1)

nad2 <- read.csv("NAD2.csv")
data_train_test_nad2 <- train_test_split(nad2, 0.7, rscu_pcg)
nad2.dt <- ml_pipeline(data_train_test_nad2,"decision_trees",split = "deviance", pretty = 1)

nad3 <- read.csv("NAD3.csv")
data_train_test_nad3 <- train_test_split(nad3, 0.7, rscu_pcg)
nad3.dt <- ml_pipeline(data_train_test_nad3,"decision_trees",split = "deviance", pretty = 1)

nad4 <- read.csv("NAD4.csv")
data_train_test_nad4 <- train_test_split(nad4, 0.7, rscu_pcg)
nad4.dt <- ml_pipeline(data_train_test_nad4,"decision_trees",split = "deviance", pretty = 1)

nad4l <- read.csv("NAD4L.csv")
data_train_test_nad4l <- train_test_split(nad4l, 0.7, rscu_pcg)
nad4l.dt <- ml_pipeline(data_train_test_nad4l,"decision_trees",split = "deviance", pretty = 1)

nad5 <- read.csv("NAD5.csv")
data_train_test_nad5 <- train_test_split(nad5, 0.7, rscu_pcg)
nad5.dt <- ml_pipeline(data_train_test_nad5,"decision_trees",split = "deviance", pretty = 1)

nad6 <- read.csv("NAD6.csv")
data_train_test_nad6 <- train_test_split(nad6, 0.7, rscu_pcg)
nad6.dt <- ml_pipeline(data_train_test_nad6,"decision_trees",split = "deviance", pretty = 1)

##### Bagging and Random Forest ######

# bagging decision trees
pcg.bag <- ml_pipeline(data_train_test,"random_forest", split = "deviance", pretty = 1, bootstraps = 10000, bagging = TRUE)

# random forest
pcg.rf <- ml_pipeline(data_train_test,"random_forest", split = "deviance", pretty = 1, bootstraps = 10000, bagging = FALSE)


##### Bagging and Random Forest ######

atp6.bag <- ml_pipeline(data_train_test_atp6,"random_forest", split = "deviance", pretty = 1, bootstraps = 10000, bagging = TRUE)
atp6.rf <- ml_pipeline(data_train_test_atp6,"random_forest", split = "deviance", pretty = 1, bootstraps = 10000, bagging = FALSE)

atp8.bag <- ml_pipeline(data_train_test_atp8,"random_forest", split = "deviance", pretty = 1, bootstraps = 10000, bagging = TRUE)
atp8.rf <- ml_pipeline(data_train_test_atp8,"random_forest", split = "deviance", pretty = 1, bootstraps = 10000, bagging = FALSE)

cox1.bag <- ml_pipeline(data_train_test_cox1,"random_forest", split = "deviance", pretty = 1, bootstraps = 10000, bagging = TRUE)
cox1.rf <- ml_pipeline(data_train_test_cox1,"random_forest", split = "deviance", pretty = 1, bootstraps = 10000, bagging = FALSE)

cox2.bag <- ml_pipeline(data_train_test_cox2,"random_forest", split = "deviance", pretty = 1, bootstraps = 10000, bagging = TRUE)
cox2.rf <- ml_pipeline(data_train_test_cox2,"random_forest", split = "deviance", pretty = 1, bootstraps = 10000, bagging = FALSE)

cox3.bag <- ml_pipeline(data_train_test_cox3,"random_forest", split = "deviance", pretty = 1, bootstraps = 10000, bagging = TRUE)
cox3.rf <- ml_pipeline(data_train_test_cox3,"random_forest", split = "deviance", pretty = 1, bootstraps = 10000, bagging = FALSE)

cob.bag <- ml_pipeline(data_train_test_cob,"random_forest", split = "deviance", pretty = 1, bootstraps = 10000, bagging = TRUE)
cob.rf <- ml_pipeline(data_train_test_cob,"random_forest", split = "deviance", pretty = 1, bootstraps = 10000, bagging = FALSE)

nad1.bag <- ml_pipeline(data_train_test_nad1,"random_forest", split = "deviance", pretty = 1, bootstraps = 10000, bagging = TRUE)
nad1.rf <- ml_pipeline(data_train_test_nad1,"random_forest", split = "deviance", pretty = 1, bootstraps = 10000, bagging = FALSE)

nad2.bag <- ml_pipeline(data_train_test_nad2,"random_forest", split = "deviance", pretty = 1, bootstraps = 10000, bagging = TRUE)
nad2.rf <- ml_pipeline(data_train_test_nad2,"random_forest", split = "deviance", pretty = 1, bootstraps = 10000, bagging = FALSE)

nad3.bag <- ml_pipeline(data_train_test_nad3,"random_forest", split = "deviance", pretty = 1, bootstraps = 10000, bagging = TRUE)
nad3.rf <- ml_pipeline(data_train_test_nad3,"random_forest", split = "deviance", pretty = 1, bootstraps = 10000, bagging = FALSE)

nad4.bag <- ml_pipeline(data_train_test_nad4,"random_forest", split = "deviance", pretty = 1, bootstraps = 10000, bagging = TRUE)
nad4.rf <- ml_pipeline(data_train_test_nad4,"random_forest", split = "deviance", pretty = 1, bootstraps = 10000, bagging = FALSE)

nad4l.bag <- ml_pipeline(data_train_test_nad4l,"random_forest", split = "deviance", pretty = 1, bootstraps = 10000, bagging = TRUE)
nad4l.rf <- ml_pipeline(data_train_test_nad4l,"random_forest", split = "deviance", pretty = 1, bootstraps = 10000, bagging = FALSE)

nad5.bag <- ml_pipeline(data_train_test_nad5,"random_forest", split = "deviance", pretty = 1, bootstraps = 10000, bagging = TRUE)
nad5.rf <- ml_pipeline(data_train_test_nad5,"random_forest", split = "deviance", pretty = 1, bootstraps = 10000, bagging = FALSE)

nad6.bag <- ml_pipeline(data_train_test_nad6,"random_forest", split = "deviance", pretty = 1, bootstraps = 10000, bagging = TRUE)
nad6.rf <- ml_pipeline(data_train_test_nad6,"random_forest", split = "deviance", pretty = 1, bootstraps = 10000, bagging = FALSE)


##### Boosting ######

library(gbm)

pcg.boost <- ml_pipeline(data_train_test,"boosting", split = "deviance", pretty = 1, bootstraps = 10000, interaction_depth = 1)

# tarefas:

# fix boosting
# do the "applied" section in page 333 with this data set
