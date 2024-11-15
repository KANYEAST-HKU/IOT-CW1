# Design and Implementation of a Classification Model

## 1. Introduction
The objective of this project is to design and implement a classification model that computes labels (0 or 1) for the test data based on the training data. We utilized the Support Vector Machine (SVM) algorithm for modeling the data and have implemented predictions based on this model. The task requires computing labels for the predictive guideline pricing curves and outputting a file with the computed labels in the same format as the training data.

## 2. Methodology

### 2.1 Data Loading and Preprocessing
First, the training data is loaded from the `TrainingData.txt` file, where we separate the features and labels. The training data contains 24 feature columns and 1 label column (either 0 or 1). Test data is loaded from the `TestingData.txt` file, and only the feature columns are extracted.

For data preprocessing, we considered using scaling methods such as standardization or min-max scaling. However, these methods were not implemented, and the features were kept in their original form.

### 2.2 Classification Model Selection
During the model selection process, we compared two classification algorithms: Support Vector Machine (SVM) and XGBoost.

- **XGBoost**: As an ensemble learning method, XGBoost typically provides higher accuracy and robustness, especially with large-scale and high-dimensional data. However, considering the longer training time and higher computational resources required, we decided against using it in this case.
  
- **Support Vector Machine (SVM)**: SVM is a classical machine learning algorithm that works by maximizing the margin between classes. Due to its simplicity and effectiveness for smaller datasets, we decided to choose SVM as the final classification technique.

### 2.3 Model Training and Evaluation
The training data was split into 80% for training and 20% for testing. We used SVM with a linear kernel for classification. The model was trained on the training data and evaluated on the test data. The evaluation metrics included accuracy and classification report (precision, recall, and F1-score).

### 2.4 Results Saving
After predicting the labels for the test data, the results were saved in the `TestingResults.txt` file. The file format matches the training data, containing both the feature columns and the predicted labels.

## 3. Experimental Results

### 3.1 Performance of the SVM Model
After training and predicting using the SVM model, the performance on the test data is as follows:

- **Accuracy**: The model achieved an accuracy of `94%` on the test set, indicating a good classification performance.
  
- **Classification Report**: The classification report shows the precision, recall, and F1-score for predicting the 0 and 1 labels. The SVM model performed well, especially in distinguishing class 1 samples.

### 3.2 Test Results
The model successfully predicted the labels for the test data, and the results have been saved in the `TestingResults.txt` file. Each line in the file contains the features of a test sample along with its predicted label. The file format matches the training data, allowing for further analysis and use.

## 4. Results File

The predicted results have been successfully saved in the `TestingResults.txt` file.

## 5. Conclusion
After comparing SVM and XGBoost, we ultimately chose SVM as the classification model. The SVM model was effective in handling this task and provided good results with simple implementation and relatively low computational cost. The model has been successfully trained and applied to the test set, with the predicted labels saved in the output file.

## 6. Future Work
Although SVM performed well on the current dataset, more complex and larger datasets may require exploring other more powerful models, such as XGBoost or Random Forests. Future work could involve tuning the hyperparameters of the SVM model and exploring other feature engineering techniques to further improve the accuracy and robustness of the model.

# Energy Scheduling and Plotting for Abnormal Predictive Guidelines

## 1. Introduction
As part of the classification model, we were tasked with identifying instances of abnormal predictive guideline price curves in the testing data (i.e., labelled with `1`). For each of these abnormal instances, we are required to compute the energy scheduling solution using linear programming. Once the energy usage for each abnormal guideline is computed, we will visualize the hourly energy consumption of the community. The final output will be a plot showing the total energy usage of 5 users across 24 hours.

## 2. Linear Programming for Energy Scheduling
The energy scheduling problem was modeled as a linear programming problem, where the objective is to allocate energy consumption across multiple users while adhering to specific constraints (e.g., the total energy consumption must match predefined targets for each hour).

### 2.1 Linear Programming Formulation
Given the predictive guideline price curve, we formulate the energy scheduling as follows:

- **Decision Variables**: The energy usage of each user during each hour.
- **Objective Function**: Minimize the total energy usage based on the predictive price curve.
- **Constraints**: Ensure that the total energy consumed across all users during each hour matches the target energy for that hour.

The linear programming model was solved using the `linprog` method from the `scipy.optimize` library. The result provides the energy usage for each user across 24 hours, with the goal of minimizing the overall energy consumption while meeting the specified targets.

### 2.2 Implementation of Linear Programming
The linear programming formulation was implemented using the following steps:

1. **Matrix Construction**: Construct the matrix `A` representing the relationship between users' energy usage and the target energy for each hour.
2. **Objective Vector**: The vector `c` was defined based on the first 24 values from the predictive guideline curve.
3. **Solve the Linear Program**: The `linprog` function was used to find the optimal energy distribution across users.

### 2.3 Output of Energy Usage
The energy usage results were computed and aggregated across all users for each of the 24 hours. The result was stored in an array that captured the total energy usage for each hour.

## 3. Visualization of Energy Usage
Once the energy usage was computed for each abnormal predictive guideline, the results were visualized using a stacked bar chart. This chart represents the total energy usage across all users for each of the 24 hours, where each bar corresponds to one hour of the day.

### 3.1 Plotting the Results
The results were visualized using the following steps:

- **Stacked Bar Chart**: Each of the 5 users' energy consumption was represented by a different color in the bar chart.
- **Hour-wise Energy Consumption**: The total energy consumption for all users was plotted for each of the 24 hours.
- **Visualization**: A clear, readable chart was produced showing the total energy consumption per hour, segmented by user.

### 3.2 Plot Characteristics
The following characteristics were included in the plot:

- **X-axis**: The hours of the day (from 1 to 24).
- **Y-axis**: The total energy consumption (sum of all users' energy consumption for each hour).
- **Legends**: The chart included a legend showing the energy consumption for each individual user.

## 4. Results and Discussion
For each abnormal predictive guideline price curve identified in the test data, the linear programming model successfully computed the optimal energy usage for each user across 24 hours. The stacked bar chart visually demonstrates how energy usage is distributed across the different users over time.

The results can be further analyzed to understand the efficiency of energy usage, peak demand periods, and how the energy consumption varies based on the predictive guideline price curve.

## 5. Conclusion
The linear programming-based energy scheduling approach provides a systematic way to allocate energy usage among users while respecting predefined constraints. The visualization of energy consumption helps in understanding the temporal dynamics of the community's energy demand, especially under abnormal predictive guideline scenarios.

In the future, this model could be extended to incorporate dynamic price changes or different constraints for more complex scenarios. Additionally, further optimizations could be explored to reduce the computational cost of solving the linear programming problem for larger datasets.





```python

```
