**By Jason Zhang**
## Framing the Problem
In this project, I will predict what caused the power outage with a classification model.
This will be a multi-class prediction problem, not a binary one, since there are seven different types of factors that caused power outages in the data.

**Why is it Important to Predict that?**
Predicting event categories causing power outages aids utilities in proactive maintenance, resource allocation, and grid resilience.
It enhances emergency response, communication, and public awareness. Compliance with regulations, data-driven
Decision-making and cost-effective planning contribute to minimizing disruptions and optimizing power infrastructure.
for societal benefit.

**Response Variable:**
I chose column `CAUSE.CATEGORY` as the response variable because it contains the primary reason for each power outage.

**Evaluation Metrics:**
Accuracy will be the most suitable metric for predicting the result since the data we predict has multiple categories.
and the distribution is balanced among all categories.
I am not using precision and recall since they are more challenging to interpret and optimize for multiclass.
classification compared to binary classification. In multiclass scenarios, you have to consider precision and recall.
for each class, which can lead to trade-offs between classes.
Thus, the accuracy of metrics is more suitable than the F-1 score, precision, and recall.

**Information Known:**
At the time of prediction, we would know the `U.S._STATE`, `CLIMATE.REGION`, `CUSTOMERS.AFFECTED`, etc. Because when a power outage happens, it's
It is obvious that we know region it happened, how many people roughly living there were affected, and the climate of that region.

**Data Cleaning:**
Done the same data cleaning and imputation processes as the last project.
For the column `CUSTOMERS.AFFECTED`, I used the RandomForestRegressor to predict the null values.

## Baseline Model
**Features Used in the Baseline Model:**

| Column                | Data Type      | Encodeing        | Description                                                                                |
|-----------------------|----------------|------------------|--------------------------------------------------------------------------------------------|
| `U.S._STATE`          | Nominal        | One-hot encoding | Represents the postal code of the U.S. states                                              |
| `OUTAGE.DURATION`     | Quantitative   | None             | Duration of outage events (in minutes)                                                     |
| `CUSTOMERS.AFFECTED`  | Quantitative   | None             | Number of customers affected by the power outage event                                     |
| `ANOMALY.LEVEL`       | Quantitative   | None             | The oceanic El Niño/La Niña (ONI) index referring to the cold and warm episodes by season. |

**Accuracy on Training Data:** 0.7539

**Accuracy on Test Data:** 0.7199

**Model Performance:**
My baseline model is not good enough since I only encoded for one column, while columns like `CUSTOMERS.AFFECTED` are
highly right-skewed, which needs some stabilization for the purpose of better prediction. Moreover, I only
included four features from the data. If we can add more correlation features to the model, our prediction could be better.

To further improve the model’s performance, additional steps can be taken, such as:
Feature engineering: explore and include additional relevant features that might have a strong correlation.
with the `CAUSE.CATEGORY`. These features could provide more discriminatory power and improve the model’s predictive ability.
Hyperparameter tuning: experiment with different hyperparameter settings for the RandomForest classification algorithms to find a configuration that works better.

## Final Model
**Features Added and New Encodeing Features in the Final Model:**

| Column              | Data Type      | Encodeing            | Description                                                                                |
|---------------------|----------------|----------------------|--------------------------------------------------------------------------------------------|
| `CLIMATE.REGION`    | Nominal        | One-hot encoding     | U.S. Climate regions as specified by National Centers for Environmental Information        |
| `TOTAL.SALES`       | Quantitative   | Quantile Transformer | Total electricity consumption in the U.S. state by Month (megawatt-hour)                   |
| `DEMAND.LOSS.MW`    | Quantitative   | None                 | Amount of peak demand lost during an outage event                                          |
| `OUTAGE.DURATION`   | Quantitative   | Standard Scaler      | Duration of outage events (in minutes) |

**Reasons of Adding New Features:**

`CLIMATE.REGION`: Power outages are heavily influenced by climate regions. Weather-related outages are more common in areas prone to severe weather,
such as hurricanes or winter storms. Furthermore, seasonal temperature variations cause distinct consumption patterns,
which would have a greater impact on the power outage. Furthermore, infrastructure adaptation and energy source mix differ by region,
which may have an impact on outage response and electricity generation.

`DEMAND.LOSS.MW`: Demand loss can be influenced by the type of power outage due to differences in the data-generating process.
For example, planned maintenance outages are typically scheduled, allowing consumers to prepare or reduce their demand temporarily.
In contrast, unexpected equipment failures or natural disasters result in unplanned outages, catching consumers off guard and leading to potentially higher demand losses.
Thus, it's considerable.

`TOTAL.SALES`: The response with higher total electricity consumption might have a higher demand for power. 
As the frequency of using power increases, the chance of encountering a power outage that is caused by
malfunctions increase.

**Algorithm Chosen:**
Use random forest classifier rather than a single decision tree classifier.

**Improved Generalization:** Random Forest reduces overfitting by averaging the predictions of multiple trees.
Each tree is trained on a random subset of the data and features, which leads to better generalization to unseen data.

**Increased Accuracy:** By aggregating the predictions of multiple trees, a random forest typically produces more accurate results than a single decision tree.
It reduces the variance associated with individual trees, resulting in a more robust model.

**Handle Complex Relationships:** Random forest can capture complex relationships in the data by combining the insights from multiple trees.
This makes it suitable for datasets with intricate patterns or interactions among features.

**Hyperparameters:**

I tuned a combination of the max depth, number of estimators, minimum sample split, and max features for the random forest to find a
combination that led to a model that generalized the best to unseen data (i.e., performs well on the test set).
The optimal hyperparameters we found were a
'max_depth': 20, 'max_features': 'auto','min_samples_split': 2, 'n_estimators': 100

**Model Performance:**

Accuracy of Training Data: 0.9571

Accuracy on Test Data: 0.8508

The accuracy of the training data increases a lot, which is reasonable as we add more features, encode, and select the best hyperparameter.
Moreover, the accuracy of test data also increases a lot, which provides a sense that the final model is much better than the baseline model.

## Fairness Analysis
We ask the question, “Does our final model perform better for power outages that happened in California or New York?
To explore this question, we will run a permutation test where we shuffle the column of `U.S._STATE`.
My evaluation metric is accuracy.

**Null Hypothesis:** Our model is fair. Its accuracy among power outages that happened in California or New York is roughly the same.
and any differences are likely due to chance.

**Alternative Hypothesis:** Our model is unfair. The difference in accuracy among power outages that happened in California or New York is statistically significant.

**Significance Level:** 5%

**p-value:** 0.634

<iframe src="pics/Fairness_Analysis.html" width=800 height=600 frameBorder=0></iframe>

After looking at the distribution and where the observed statistic lied, we saw that the result was not statistically significant.
with a p-value of 0.634, greater than 0.01 (our significance level), thus we fail to reject the null hypothesis.
This suggests our model is possibly fair for predicting what caused the power outages that happened in California and New York.

