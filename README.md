# Predicting MLB Gold Glove Winners with Python
![pandas](https://img.shields.io/badge/pandas-blue) ![numpy](https://img.shields.io/badge/numpy-orange) ![scikit-learn](https://img.shields.io/badge/scikit--learn-green) ![matplotlib](https://img.shields.io/badge/matplotlib-yellow)

This project aims to predict the winners of the MLB Gold Glove Award across all positions in both the American and National Leagues by leveraging position-specific and league-specific defensive statistics. The project utilizes machine learning techniques to rank players based on their likelihood of winning the award, providing valuable insights for baseball scouts, analysts, teams, and fans.

## Problem and Relevance

The MLB Gold Glove Award annually recognizes the best defensive players at their respective positions. The problem involves:
- Analyzing complex fielding metrics to identify defensive excellence.
- Addressing imbalanced data where only ~1% of entries represent winners.
- Customizing predictions based on positional and league-specific characteristics.

This predictive model can aid in player scouting, development, and strategic decision-making for teams.

## Datasets

### Data Sources
- **Baseball-Reference**: Collected detailed fielding metrics for all MLB players (2013–2024).
- **Player Datasets**: Individual player fielding performance metrics for each position, including:
  - Games (G), Innings (Inn), Putouts (PO), Assists (A), Errors (E), etc.
- **League Datasets**: Average fielding metrics for the National and American Leagues.

### Dataset Size
- **Player Dataset**: 30,000+ rows spanning 12 seasons.
- **League Dataset**: 24 rows per position for league averages.

### Target Variable
- `Win`: Binary variable indicating whether the player won the Gold Glove Award (1 for winners, 0 otherwise).

### Data Cleaning and Preprocessing
- Removed rows with missing or null values.
- Handled duplicate player entries (e.g., trades or free agency).
- Normalized player metrics using league averages to account for seasonal and league-specific trends.

## Machine Learning Models

### Models Utilized
1. Random Forest
2. Logistic Regression
3. Gradient Boosting
4. k-Nearest Neighbors (kNN)
5. Support Vector Machine (SVM)

### Experimental Setup
- **Training Data**: 2013–2023 seasons.
- **Test Data**: 2024 season.

### Hyperparameter Tuning
- Optimized parameters using `GridSearchCV`.

### Evaluation Metric
- **AUC-ROC**: Chosen to address the class imbalance and focus on the model's ability to distinguish winners.

## Key Results

### Model Performance (AUC-ROC Scores)
- **Random Forest**: Up to 0.9966 for certain positions.
- **Gradient Boosting**: Strong results across all positions (e.g., 0.9944 for Shortstop).
- **kNN**: Perfect score for Second Base (1.0000).
- **SVM**: Top performer for several positions (1.0000 for Third Base).

## Challenges and Opportunities

### Challenges
1. **Class Imbalance**: Majority of players are non-winners, leading to high recall for class 0 but poor recall for class 1.
2. **Subjective Criteria**: Gold Glove selections are influenced by subjective voting criteria beyond fielding metrics.

### Opportunities for Improvement
1. **Incorporate Pre-2013 Data**: Extend the dataset by including earlier seasons to increase observations for winners.
2. **Filter Eligible Players**: Ensure only players meeting award qualifications are included (e.g., minimum innings played).

## References

[Machine Learning Model Predicting MLB Gold Glove Award Winners](https://github.com/lucaskelly49/Machine-Learning-Model-Predicting-MLB-Gold-Glove-Award-Winners)

A guidance resource for a machine learning project developed as part of the Module 3 Final Project for a Flatiron School curriculum

---

This project was completed as part of the **QTM 347 Machine Learning I course** at Emory University under the guidance of **Professor Ruoxuan Xiong**. The datasets for this project were sourced and curated using MLB stats and records from **Baseball-Reference.com**.
