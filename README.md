# Instagram Analytics Project

## Introduction
This project aims to analyze Instagram post data to gain insights into various factors affecting post impressions, engagement, and conversion rates. The analysis includes visualizations using popular Python libraries like Pandas, Matplotlib, Seaborn, Plotly Express, and WordCloud. Additionally, machine learning techniques are employed to predict impressions based on post metrics.

## Project Structure
- `Instagram.csv`: Dataset containing Instagram post data.
- `Instagram_Analytics.ipynb`: Jupyter Notebook containing the Python code for data analysis and visualization.
- `README.md`: Documentation file providing an overview of the project, installation instructions, and usage guide.

## Requirements
To run this project, ensure you have the following dependencies installed:
- pandas
- numpy
- matplotlib
- seaborn
- plotly
- wordcloud
- scikit-learn

You can install these dependencies using pip:
pip install pandas numpy matplotlib seaborn plotly wordcloud scikit-learn

## Analysis Overview
1. **Data Loading and Preprocessing**: Load the Instagram dataset, handle missing values, and perform basic data exploration.
2. **Visualizations**:
    - Distribution of impressions from different sources (home, hashtags, explore).
    - Pie chart showing the proportion of impressions from various sources.
    - WordCloud visualization of the most used hashtags.
    - Scatter plots depicting the relationship between likes, comments, shares, saves, and impressions.
3. **Correlation Analysis**: Calculate correlation coefficients between different post metrics and impressions.
4. **Conversion Rate Calculation**: Calculate the conversion rate based on the number of follows and profile visits.
5. **Machine Learning Model**: Train a Passive Aggressive Regressor model to predict impressions based on post metrics.
6. **Prediction Example**: Demonstrate how to use the trained model to predict impressions for a given set of post metrics.

## Contributors
- Parsabzh

## License
This project is licensed under the [MIT License](LICENSE).
