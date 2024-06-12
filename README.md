# MARKET-MIND-ML-POWERED-STOCK-TREND-TRACKER
Jaypee Institute of Information Technology (Major Project)

August 2023-May 2024

Overview:
This project represents a comprehensive investigation into the realm of stock market prediction, where
we deploy a diverse array of machine learning (ML) algorithms and neural network architectures.

Objective:
The project's objective is to investigate various forecasting methods for estimating future stock. We do
this by analysing the ostensibly chaotic market data and use supervised learning techniques for stock
price predictions.

Methodology:
The methodology for the stock market prediction project involved a structured approach beginning with
data extraction using the yfinance library and subsequent preprocessing to ensure data integrity.
Following this, data scaling was conducted using the MinMaxScaler function from scikit-learn, with a
70-30 split for training and testing sets, preserving temporal order for realism. Nine diverse models
were then evaluated, ranging from unconventional optimization algorithms like Ant Colony
Optimization (ACO), Particle Swarm Optimization (PSO), Differential Evolution (DE), and Artificial
Bee Colony (ABC), to various neural network architectures including Artificial Neural Networks
(ANN), Recurrent Neural Networks (RNN), Long Short-Term Memory (LSTM), and Convolutional
Neural Networks (CNN). Additionally, a hybrid CNN-LSTM model was introduced, capitalizing on the
strengths of both architectures to enhance predictive capabilities. Evaluation metrics such as accuracy,
F1 score, and R-squared score were employed to assess model performance on the testing dataset,
ultimately facilitating the identification of the most proficient predictive model for stock market
forecasting.
In the final phase, the hybrid CNN-LSTM model emerged as a novel approach, integrating both spatial
feature extraction from CNNs and temporal dependency modeling from LSTMs. This hybrid
architecture demonstrated improved predictive capabilities compared to individual models,
underscoring its effectiveness in capturing intricate patterns within the stock market data. Evaluation
involved comprehensive analysis using established metrics, affirming the hybrid model's superiority in
stock market prediction. By leveraging cutting-edge methodologies and diverse modeling techniques,
the project aimed to push the boundaries of predictive modeling in financial markets, offering valuable
insights for investors and traders alike.

Results:
The process of training machine learning models involves several crucial steps. Initially, continuous
features are normalized to ensure consistent scaling across variables. Subsequently, the primary dataset
is randomly divided into training and test sets, with 30% reserved for testing to assess model
performance on unseen data. Models are then trained on the training set and evaluated using validation
data, employing techniques like 'early stopping' to prevent overfitting by halting training when
performance on the validation set deteriorates. In the context of deep learning models, input values are
reshaped to three dimensions, representing (samples, time steps, and features). Techniques like weight
regularization and dropout layers are incorporated to further combat overfitting. The entire process is
implemented in Python3, utilizing libraries such as Scikit Learn and Keras to facilitate efficient coding
and experimentation.
It can be seen that ACO, PSO, DE and ABC are least accurate(approximately 47.8% F1-Score and
73.9% R2-Score) while LSTM and CNN are top predictors (roughly 98% F1-Score and 98.2% R2-
Score) with a considerable difference compared to other models. As a result, a hybrid model combining
elements of both CNN and LSTM was developed, resulting in an unprecedented perfect (99.99% F1-
Score and R2-Score). This hybrid approach leverages the strengths of both architectures to further
enhance predictive capabilities, underscoring its potential for superior performance in stock market
prediction tasks.

Technologies Used:
- Programming Languages: Python
- Libraries/Frameworks: pandas, NumPy, scikit-learn, TensorFlow/Keras (for deep learning)
- Data Visualization: Matplotlib

Lessons Learned:
- Importance of feature selection and engineering in improving model performance.
- Understanding the limitations of predictive models in highly unpredictable markets.

Future Enhancements:
- Integration of alternative data sources (e.g., social media sentiment, satellite imagery) for improved
predictions.
- Implementation of ensemble learning techniques to further enhance model robustness.
- Development of a comprehensive risk management module to accompany the prediction model.

Conclusion:
This project showcases a successful application of machine learning techniques in predicting stock
market trends. By leveraging historical data and advanced algorithms, the developed model offers
valuable insights for investors and traders, potentially aiding in more informed decision-making and
risk management.
