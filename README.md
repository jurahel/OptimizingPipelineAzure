# Optimizing an ML Pipeline in Azure


***
## Overview
I worked on a project for the Udacity Azure ML Nanodegree. In it, I created and improved a pipeline using Azure ML and the Python SDK. I built a custom model using Scikit-learn Logistic Regression and fine-tuned its hyperparameters with HyperDrive. Additionally, I used Azure AutoML to find an optimal model with the same data, allowing me to compare the results of both methods. 
First, I set up a train script and analyzed a dataset using a custom Scikit-learn logistic regression model. In Step 2, I created a Jupyter Notebook to find the best hyperparameters for the model using HyperDrive. In Step 3, I loaded the same dataset in the notebook and used AutoML to find another optimized model. Lastly, in Step 4, I compared the results of both methods and summarized my findings in a research report, which you can find in this Readme file.

***
## Summary
This dataset is about people and their response to marketing calls from a Portuguese bank. The goal is to predict if a person will subscribe to a bank term deposit. The top-performing model was the AutoML Voting Ensemble with an accuracy of 0.91642, derived from a Scikit-learn pipeline. In comparison, the HyperDrive model had an accuracy of 0.9176024. So a minor difference in favor of the Hyper drive Model.

## Scikit-learn Pipeline
Firstly, for parameter sampling, I utilized RandomParameterSampling to explore discrete values for regularization (C) and maximum iterations (max_iter). This choice, involving the choice function, allowed me to select specific values for both parameters.
```
ps = RandomParameterSampling(
    {
        '--C' : choice(0.001,0.01,0.1,1,10,20,50,100,200,500,1000),
        '--max_iter': choice(50,100,200,300)
    }
)
```
I chose discrete values with _choice_ for both parameters, _C_ and _max_iter_ and _C_ is the Regularization while _max_iter_ is the maximum number of iterations.

Moving on to the early stopping policy, I opted for BanditPolicy, specifying evaluation intervals and a slack factor. The evaluation interval represents the frequency for applying the policy, while the slack factor determines the allowed slack with respect to the best-performing training run. This policy automatically terminates poorly performing runs, contributing to improved computational efficiency.

Now, for AutoML, I configured the run with specific parameters. The experiment has a 15-minute timeout criterion, indicating how long the experiment should run. The task is classification, and the primary metric chosen is accuracy. The training data comes from a specified dataset, and the label column is 'y'. Enabling ONNX-compatible models adds flexibility, and I set the number of cross-validations to 2 for robustness.

In summary, this pipeline involves thoughtful choices in parameter sampling, an effective early stopping policy, and a well-configured AutoML run to generate models with their corresponding hyperparameters.

## Pipeline comparison

As mentioned earlier, the discrepancy in results is minor. Given additional time for the AutoML process, the resulting model would likely show significant improvement. The notable advantage lies in AutoML's ability to handle all the essential calculations, training, validations, and more autonomously, eliminating the need for manual adjustments or iterations. This sets it apart from the Scikit-learn Logistic Regression pipeline, where we have to actively make adjustments, changes, and undergo numerous trials and errors to arrive at a final model. 

## Future work
There are a couple of areas where our future experiments could see notable improvements. Firstly, addressing the issue of highly imbalanced data is crucial. The imbalance, as shown in the plot, can mislead accuracy metrics, particularly favoring the majority class. To mitigate this, alternative metrics like AUC_weighted are more suitable for imbalanced data. Additionally, exploring different algorithms or employing techniques such as random under-sampling of the majority class or random over-sampling of the minority class can be beneficial. The imbalanced-learn package offers further tools for handling this issue.

Another aspect for enhancement is the parameter for cross-validations (n_cross_validations). While increasing this parameter generally leads to higher accuracy, it also extends computation time and costs. Striking a balance between accuracy and computational efficiency is essential. If the number of cross-validations is adjusted, it's important to consider extending the experiment timeout accordingly, as the current setting of 15 minutes may not suffice. These improvements, especially addressing data imbalance and optimizing cross-validations, can significantly enhance the model's performance in future executions.
