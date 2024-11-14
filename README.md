# Cryptocurrency-Price-Prediction-ML-Model-📈
![image](https://github.com/user-attachments/assets/185d3abf-8581-44e8-93fc-839d2c75d262)

## Overview
This project aims to predict the price of Ethereum, one of the most popular cryptocurrencies, using machine learning techniques. By analyzing historical price data and integrating various technical indicators, the model seeks to provide insights and forecasts on future price movements, which can assist investors and traders in making data-driven decisions.

## Key Findings
- Price Volatility: Ethereum's price shows significant fluctuations, emphasizing the need for a predictive model that accounts for volatility.
- Correlation with Volume: High trading volumes often correlate with price spikes, which the model uses as a feature to enhance prediction accuracy.
- Technical Indicators: Features like moving averages, volatility metrics, and lagged values provide valuable insights into market trends, improving model performance.
- Prediction Accuracy: After tuning and testing, the model achieves an optimal Mean Squared Error (MSE) of approximately 0.02 on the test set, making it a reliable tool for short-term forecasting.

## Visualizations
- The project includes various visualizations to support the findings and illustrate the model's accuracy:
- Price Trends: Line plots show historical prices and predictions over time.
- Correlation Matrix: A heatmap displaying correlations between features, identifying relationships like volume and volatility.
- Residual Plot: To visualize the errors in predictions, highlighting areas for potential improvement.
- Prediction vs Actual: Scatter and line plots to compare predicted prices with actual values, giving a clear view of model performance.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib & Seaborn (for data visualization)

## Data Preprocessing
Handling Missing Values: Missing entries are filled with forward-fill or mean substitution methods.
Scaling: Prices and volume data are normalized for more stable model training.
Feature Engineering: New features such as moving averages, volatility indicators, and lagged price data are generated to capture more predictive patterns.
Data Splitting: Data is divided into training, validation, and test sets, ensuring the model is evaluated effectively.

## Modeling
The project explores multiple machine learning models to find the best fit:
Linear Regression: Provides a baseline prediction.
Random Forest Regressor: Captures non-linear patterns and interactions between features.
Long Short-Term Memory (LSTM): A neural network architecture designed for sequential data, useful for predicting time-series data like cryptocurrency prices.

## Evaluation Metrics
- To assess the model's performance, we use:

- Mean Squared Error (MSE): Measures the average squared differences between predicted and actual prices.
- Root Mean Squared Error (RMSE): The square root of MSE, providing error in actual units.
- Mean Absolute Percentage Error (MAPE): Indicates the accuracy as a percentage.

Getting Started
##  Clone the repository:

bash
   - git clone https://github.com/UdayKiranVanapalli/crypto-price-prediction.git


- The model's predictions align well with actual Ethereum prices on the test data, demonstrating strong performance in short-term forecasting. Final metrics and visualizations show a promising level of accuracy, although there are limitations due to market volatility and external economic factors.

## Future Work
- Additional Features: Integrate more macroeconomic indicators, like interest rates and inflation, to improve model robustness.
- Hyperparameter Tuning: Experiment with more advanced hyperparameter optimization techniques, such as GridSearchCV or Bayesian optimization.
- Ensemble Models: Combine predictions from multiple models to achieve better stability and performance.

## Conclusion
This project provides a strong foundation for cryptocurrency price prediction. It highlights the importance of feature engineering and model selection in accurately forecasting highly volatile markets like cryptocurrency. Future work can build on this framework to enhance prediction reliability further.

## Contributing
Contributions are welcome! If you'd like to improve the model or add features, please feel free to open an issue or submit a pull request.
