# 🍄 ShroomBoom - Mushroom Classification Application

A comprehensive Streamlit web application that predicts whether a mushroom is edible or poisonous using machine learning models.

## Features

### Home Page

- Dataset overview with key statistics
- Safety warnings for mushroom identification
- Visual distribution of edible vs poisonous mushrooms

### Model Prediction

- Interactive form to input mushroom characteristics
- Support for multiple ML models (Logistic Regression, Random Forest, SVM)
- Real-time predictions with confidence scores
- Visual confidence indicators

### Visualizations

- Feature distribution analysis by class
- Interactive correlation heatmap
- Customizable data exploration tools

### Model Performance

- Accuracy comparison across all models
- Feature importance analysis
- Model evaluation metrics

### Help & Documentation

- Comprehensive feature descriptions
- Usage instructions
- Safety guidelines for mushroom identification

## Technical Features

- **Error Handling**: Comprehensive error handling with user-friendly messages
- **Loading States**: Loading indicators for better user experience
- **Responsive Design**: Clean, professional interface that works on different screen sizes
- **Caching**: Optimized performance with Streamlit caching for data and models
- **Interactive Elements**: Dynamic visualizations using Plotly and Seaborn

## Installation & Setup

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd ShroomBoom
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**

   ```bash
   streamlit run app.py
   ```

4. **Access the application**
   - Open your browser and navigate to `http://localhost:8501`

## Required Files

The application expects the following files in the project directory:

- `mushrooms.csv` - The mushroom dataset
- `mushroom_lr_pipeline.joblib` - Logistic Regression model
- `mushroom_rf_pipeline.joblib` - Random Forest model
- `mushroom_svm_pipeline.joblib` - SVM model
- `metadata.joblib` - Feature metadata

## Dependencies

- streamlit==1.28.1
- pandas==2.1.1
- numpy==1.26.4
- scikit-learn==1.6.1
- plotly==5.17.0
- seaborn==0.12.2
- matplotlib==3.7.2
- joblib==1.3.2

## Dataset Features

The application uses 22 mushroom characteristics for classification:

1. **Cap features**: shape, surface, color
2. **Physical properties**: bruises, odor
3. **Gill characteristics**: attachment, spacing, size, color
4. **Stalk features**: shape, root, surface, color
5. **Veil properties**: color
6. **Ring features**: number, type
7. **Spore characteristics**: print color
8. **Growth patterns**: population, habitat

## Safety Warning

**IMPORTANT**: This application is for educational and research purposes only.

**NEVER consume wild mushrooms** based solely on this prediction tool or any automated system. Mushroom identification requires expert knowledge, and misidentification can result in serious illness or death.

Always consult with mycology experts and use multiple reliable sources for mushroom identification.

## Model Information

The application includes three trained machine learning models:

1. **Logistic Regression**: Fast, interpretable linear model
2. **Random Forest**: Ensemble method with feature importance
3. **Support Vector Machine**: Robust classification with kernel tricks

All models are pre-trained on the mushroom dataset and packaged as scikit-learn pipelines for easy deployment.

## Usage Tips

1. **Navigation**: Use the sidebar to switch between different sections
2. **Predictions**: Fill out all mushroom characteristics for best results
3. **Model Selection**: Try different models to compare predictions
4. **Visualizations**: Explore data patterns to understand mushroom characteristics
5. **Performance**: Check model accuracy to understand reliability

## Contributing

Feel free to contribute by:

- Reporting bugs
- Suggesting new features
- Improving documentation
- Adding new visualization types

## License

This project is for educational purposes. Please ensure proper attribution when using the code or models.
