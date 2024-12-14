# Predicting MLB Gold Glove Winners with Python
![pandas](https://img.shields.io/badge/pandas-blue) ![numpy](https://img.shields.io/badge/numpy-orange) ![scikit-learn](https://img.shields.io/badge/scikit--learn-green) ![matplotlib](https://img.shields.io/badge/matplotlib-yellow)

This project aims to **predict the winners of the Rawlings Gold Glove Award across all positions in both the American and National Leagues** by leveraging position-specific and league-specific defensive statistics. The project utilizes machine learning techniques to rank players based on their likelihood of winning the award, providing valuable insights for baseball scouts, analysts, teams, and even fans.

Checkout the **[Jupyter Notebook](./final-code.ipynb)** for code and **[Presentation Slides](./presentation-slides.pdf)** for an overview of the project!

## Problem and Relevance

The **motivation behind this project** is twofold.

1. Defensive performance is a critical but often undervalued aspect of player evaluation in baseball. Accurately predicting Gold Glove winners can provide teams with actionable insights for scouting, player development, and strategic planning.

2. This project demonstrates the application of machine learning to a challenging real-world problem, showcasing its potential in sports analytics. Beyond baseball, the methodologies employed here could be applied to other fields where prediction tasks involve imbalanced datasets or subjective outcomes.

The Rawlings Gold Glove Award annually recognizes the best defensive players at their respective positions. The problem involves:
- Understanding complex fielding data and translating them into predictive models that can highlight defensive excellence.
- Addressing imbalanced data where only ~1% of entries represent targets.
- Customizing predictions based on positional and league-specific characteristics.

## Approaches

### Initial Objective: Deterministic Classification
- Predicts winners as a binary outcome: 1 (Winner) or 0 (Non-Winner).  
- **Issue**: Class imbalance (only ~1% of data represent winners).
  - Models predominantly predicted 0, resulting in high accuracy but poor utility for predicting winners.

### New Objective: Probabilistic Classification
- Predicts probabilities rather than binary labels.
- Provides more nuanced insights to rank players based on their likelihood of winning.
- Highlights strong candidates, even if they are not predicted winners.

## Data Overview

### Data Collection
Collected fielding metrics for all MLB players across 9 positions from **Baseball-Reference.com**.
- Pitcher, Catcher, First Base, Second Base, Third Base, Shortstop, Left Field, Center Field, Right Field
- Data only from 2013 to 2024, as fielding metrics became more comprehensive starting in 2013.

Created separate player and league datasets for each position.
- **Player Datasets**: Individual player fielding performance metrics for each position, including:
  - Games (G), Innings (Inn), Putouts (PO), Assists (A), Errors (E), etc.
- **League Datasets**: Average fielding metrics for the National and American Leagues by season.

Added the following columns for the classification task.
- Season: The year in which the player's fielding performance is recorded.
- Champion: If the player’s team won the league championship of that season.
- **Win**: If the player won the Gold Glove Award for that position.

### Data Size
- **Player Datasets**: 30,000+ rows spanning 12 seasons.
- **League Datasets**: 24 rows per position representing league averages for 12 seasons.
- **Basic statistics** are available in the **[Jupyter Notebook](./final-code.ipynb)**.

### Target Variable
- **Win**: Binary variable indicating whether the player won the Gold Glove Award (1 for winners, 0 otherwise).

### Data Cleaning and Preprocessing
Removed rows with missing or null values.
- The fielding metrics are specific to each player's performance for a particular season.
- Using mean or median values would introduce artificial data that does not accurately reflect the individual player's abilities or contributions to defense.

Handled duplicate player entries (e.g. trades or free agency) by manually assigning teams.
- Prioritized selecting the team with the most innings played, followed by the most games played, and followed by the last team listed in the record.

Dropped fielding metric columns not available in both player and league datasets for normalization.

Normalized player metrics using league averages to account for seasonal and league disparities.
- Fielding performance might have improved in recent years (e.g. better equipment, analytics).
- Metrics might vary between the National and American Leagues.
- Retained the original value for normalization where division by zero occurred.

## Modeling

### Machine Learning Models
1. Random Forest
2. Logistic Regression
3. Gradient Boosting
4. k-Nearest Neighbors (kNN)
5. Support Vector Machine (SVM)

### Experimental Setup
- **Training Data (2013–2023)**: Used data from previous 11 seasons to train the models.
- **Test Data (2024)**: Reserved the most recent season’s data to evaluate real-world predictive performance.

### Hyperparameter Tuning
Optimized parameters using `GridSearchCV`.
- Random Forest: n_estimators, max_depth, min_samples_split, min_samples_leaf, bootstrap
- Logistic Regression: penalty, max_iter
- Gradient Boosting: n_estimators, learning_rate, max_depth, min_samples_split, min_samples_leaf
- kNN: n_neighbors, weights, power parameter (p)
- SVM: kernel, regularization parameter (C), gamma

### Evaluation Metric
- **AUC-ROC**: Chosen to address the class imbalance and better reflect the model's performance on the minority class (winners).
- Unlike accuracy, which can be skewed by the majority class, AUC-ROC focuses on the model's ability to distinguish between classes.

## Key Results

### Model Performance by Position (AUC-ROC Scores)

**First Base**
  - Random Forest: 0.8854
  - Gradient Boosting: 0.8646
  - Logistic Regression: 0.9583
  - **kNN: 0.9792**
  - SVM: 0.8958

**Second Base**
  - Random Forest: 0.9966
  - Gradient Boosting: 0.9966
  - Logistic Regression: 0.9832
  - **kNN: 1.0000**
  - SVM: 0.9732

**Shortstop**
  - Random Forest: 0.9775
  - Gradient Boosting: 0.9685
  - Logistic Regression: 0.9640
  - kNN: 0.7095
  - **SVM: 0.9820**

**Third Base**
  - Random Forest: 0.9889
  - **Gradient Boosting: 0.9944**
  - Logistic Regression: 0.9333
  - kNN: 0.7361
  - SVM: 0.9333

**Catcher**
  - **Random Forest: 0.9866**
  - Gradient Boosting: 0.9677
  - Logistic Regression: 0.9032
  - kNN: 0.7151
  - SVM: 0.9086

**Pitcher**
  - **Random Forest: 0.9792**
  - Gradient Boosting: 0.9626
  - Logistic Regression: 0.9618
  - kNN: 0.4942
  - SVM: 0.9776

**Left Field**
  - Random Forest: 0.9831
  - Gradient Boosting: 0.9758
  - Logistic Regression: 0.9976
  - kNN: 0.4952
  - **SVM: 1.0000**

**Center Field**
  - Random Forest: 0.9483
  - Gradient Boosting: 0.9379
  - Logistic Regression: 0.8966
  - **kNN: 0.9828**
  - SVM: 0.9276

**Right Field**
  - **Random Forest: 0.9924**
  - Gradient Boosting: 0.9836
  - Logistic Regression: 0.9646
  - kNN: 0.7475
  - SVM: 0.9520

## Discussion and Conclusion

Overall, the models showed **strong ranking ability across positions, as evidenced by high AUC-ROC scores**, but failed to predict 2024 winners due to the lack of sufficient training examples for class 1. This indicates that the models were effective at identifying top candidates for the award but struggled to learn patterns that accurately classify the minority class (winners). Across all models, accuracy was high, reflecting the dominance of non-winners in the datasets.

### Challenges
1. **Class Imbalance**: With winners making up only ~1% of the datasets, the models prioritized predicting the majority class (non-winners), leading to perfect recall for class 0 but zero recall for class 1.
2. **Subjective Voting Criteria**: Gold Glove decisions are influenced by subjective (external) factors beyond fielding metrics. As a result, some players with high probabilistic scores were not selected as winners, and vice versa.

### Future Directions
1. **Pre-2013 Data**: Include data from seasons prior to 2013, even if it requires discarding unavailable metrics, to improve the representation of the minority class (winners).
2. **Selection Criteria**: Filter datasets to include only players meeting minimum playing time requirements introduced in 2022, aligning the data more closely with real-world award criteria.
3. **Subjective Factors**: Explore incorporating qualitative data, such as media reports or fan sentiment, to account for the subjective factors that influence Gold Glove Award decisions.

## References

[Machine Learning Model Predicting MLB Gold Glove Award Winners](https://github.com/lucaskelly49/Machine-Learning-Model-Predicting-MLB-Gold-Glove-Award-Winners)

A guidance resource for a machine learning project developed as part of the Module 3 Final Project for a Flatiron School curriculum

---

This project was completed as part of the **QTM 347 Machine Learning I course** at Emory University under the guidance of **Professor Ruoxuan Xiong**. The datasets for this project were sourced and curated using MLB stats and records from **Baseball-Reference.com**.
