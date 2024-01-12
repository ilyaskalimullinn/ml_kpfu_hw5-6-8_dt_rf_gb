# Decision Trees, Random Forest, Gradient Boosting

## Decision Tree

Build decision tree for classification task and train it on digits dataset. 
- Divide dataset on train, test, valid sets
- Splitting function — is a hyperplane parallel to the coordinate of the axis
- Build terminal node if one of the three condition is true
  - The depth of the tree is bigger than some threshold
  - The entropy of the targets, that came to the node is smaller than some threshold
  - Number of elements, that came to the node is lower than some threshold
- Calculate accuracy and build confusion matrix for validation and test sets after building decision tree


Build decision tree for regression task and train it on wine quality dataset. 
- Splitting function — is a hyperplane parallel to the coordinate of the axis
- Build terminal node if one of the three condition is true
  - The depth of the tree is bigger than some threshold
  - The entropy of the targets, that came to the node is smaller than some threshold
  - Number of elements, that came to the node is lower than some threshold
- Calculate error(mentioned in practise) value on validation and test sets after building decision tree.

## Random Forest

- Build Random Forest for classification task and train it on digits dataset. 
- Divide dataset on train, test, valid sets
- Before training use standardisation preprocessing. 
- Validate values of  M,  L_1 (max_nb_dim_to_check),  L_2 (max_nb_thresholds). Use Radom Search method. Train at leat 30 models
- Find 10 best models according to accuracy on validation set. Make plot with 10 points, where x-axis name of the model ( ,  ,   values) , y-axis accuracy on valid set. Add to hover_data accuracy on test set
- Build confusion matrix for best model on test set

Bonus 
- Train Random Forest using bagging

## Gradient Boosting

- Build gradient boosted decision trees for regression model and train it on wine dataset.
- Validate values of weak_learners_amount, learning_rate. Use Random Search method. Train at least 30 models.
- Find 10 best models according to accuracy on validation set. Make plot with 10 points, where x-axis name of the model ( ,  ,   values) , y-axis accuracy on valid set. Add to hover_data accuracy on test set

## Some visualisations

![image](https://github.com/ilyaskalimullinn/ml_hw5_decision_trees/assets/90423658/234d2c88-d4ae-44cb-bf58-6fb638f6752b)

![image](https://github.com/ilyaskalimullinn/ml_hw5_decision_trees/assets/90423658/be8c9ef9-bae6-4af5-907a-f43dd494ba03)

![image](https://github.com/ilyaskalimullinn/ml_hw5_decision_trees/assets/90423658/aad5530a-cb7d-4fcf-aca5-b1eb1b9ff468)

