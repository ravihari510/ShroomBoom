import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="ShroomBoom - Mushroom Classifier",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .danger {
        color: #ff4b4b;
        font-weight: bold;
    }
    .safe {
        color: #00d4aa;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the mushroom dataset"""
    try:
        data = pd.read_csv('mushrooms.csv')
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def load_models_and_metadata():
    """Load all trained models and metadata"""
    try:
        models = {}
        models['Logistic Regression'] = joblib.load('mushroom_lr_pipeline.joblib')
        models['Random Forest'] = joblib.load('mushroom_rf_pipeline.joblib')
        models['SVM'] = joblib.load('mushroom_svm_pipeline.joblib')
        
        metadata = joblib.load('metadata.joblib')
        
        return models, metadata
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

def get_feature_mappings():
    """Get feature value mappings for user-friendly display"""
    return {
        'cap-shape': {
            'b': 'Bell', 'c': 'Conical', 'x': 'Convex', 'f': 'Flat',
            'k': 'Knobbed', 's': 'Sunken'
        },
        'cap-surface': {
            'f': 'Fibrous', 'g': 'Grooves', 'y': 'Scaly', 's': 'Smooth'
        },
        'cap-color': {
            'n': 'Brown', 'b': 'Buff', 'c': 'Cinnamon', 'g': 'Gray',
            'r': 'Green', 'p': 'Pink', 'u': 'Purple', 'e': 'Red',
            'w': 'White', 'y': 'Yellow'
        },
        'bruises': {
            't': 'True', 'f': 'False'
        },
        'odor': {
            'a': 'Almond', 'l': 'Anise', 'c': 'Creosote', 'y': 'Fishy',
            'f': 'Foul', 'm': 'Musty', 'n': 'None', 'p': 'Pungent', 's': 'Spicy'
        },
        'gill-attachment': {
            'a': 'Attached', 'd': 'Descending', 'f': 'Free', 'n': 'Notched'
        },
        'gill-spacing': {
            'c': 'Close', 'w': 'Crowded', 'd': 'Distant'
        },
        'gill-size': {
            'b': 'Broad', 'n': 'Narrow'
        },
        'gill-color': {
            'k': 'Black', 'n': 'Brown', 'b': 'Buff', 'h': 'Chocolate',
            'g': 'Gray', 'r': 'Green', 'o': 'Orange', 'p': 'Pink',
            'u': 'Purple', 'e': 'Red', 'w': 'White', 'y': 'Yellow'
        },
        'stalk-shape': {
            'e': 'Enlarging', 't': 'Tapering'
        },
        'stalk-root': {
            'b': 'Bulbous', 'c': 'Club', 'u': 'Cup', 'e': 'Equal',
            'z': 'Rhizomorphs', 'r': 'Rooted', '?': 'Missing'
        },
        'stalk-surface-above-ring': {
            'f': 'Fibrous', 'y': 'Scaly', 'k': 'Silky', 's': 'Smooth'
        },
        'stalk-surface-below-ring': {
            'f': 'Fibrous', 'y': 'Scaly', 'k': 'Silky', 's': 'Smooth'
        },
        'stalk-color-above-ring': {
            'n': 'Brown', 'b': 'Buff', 'c': 'Cinnamon', 'g': 'Gray',
            'o': 'Orange', 'p': 'Pink', 'e': 'Red', 'w': 'White', 'y': 'Yellow'
        },
        'stalk-color-below-ring': {
            'n': 'Brown', 'b': 'Buff', 'c': 'Cinnamon', 'g': 'Gray',
            'o': 'Orange', 'p': 'Pink', 'e': 'Red', 'w': 'White', 'y': 'Yellow'
        },
        'veil-color': {
            'n': 'Brown', 'o': 'Orange', 'w': 'White', 'y': 'Yellow'
        },
        'ring-number': {
            'n': 'None', 'o': 'One', 't': 'Two'
        },
        'ring-type': {
            'c': 'Cobwebby', 'e': 'Evanescent', 'f': 'Flaring',
            'l': 'Large', 'n': 'None', 'p': 'Pendant', 's': 'Sheathing', 'z': 'Zone'
        },
        'spore-print-color': {
            'k': 'Black', 'n': 'Brown', 'b': 'Buff', 'h': 'Chocolate',
            'r': 'Green', 'o': 'Orange', 'u': 'Purple', 'w': 'White', 'y': 'Yellow'
        },
        'population': {
            'a': 'Abundant', 'c': 'Clustered', 'n': 'Numerous',
            's': 'Scattered', 'v': 'Several', 'y': 'Solitary'
        },
        'habitat': {
            'g': 'Grasses', 'l': 'Leaves', 'm': 'Meadows',
            'p': 'Paths', 'u': 'Urban', 'w': 'Waste', 'd': 'Woods'
        }
    }

def show_help():
    """Display help information"""
    st.markdown("""
    ## 🍄 ShroomBoom Help Guide
    
    ### How to Use This Application:
    
    1. **Navigation**: Use the sidebar to navigate between different sections
    2. **Model Prediction**: 
       - Select mushroom characteristics from the dropdown menus
       - Choose which model to use for prediction
       - Click "Predict" to get results
    3. **Visualizations**: Explore data patterns and distributions
    4. **Model Performance**: View accuracy metrics for all models
    
    ### Feature Descriptions:
    - **Cap Shape/Surface/Color**: Physical appearance of the mushroom cap
    - **Bruises**: Whether the mushroom shows bruising when handled
    - **Odor**: Smell characteristics (very important for safety!)
    - **Gill Features**: Characteristics of the gills under the cap
    - **Stalk Features**: Properties of the mushroom stem
    - **Veil/Ring**: Presence and type of veil or ring structures
    - **Spore Print**: Color of spores when printed
    - **Population/Habitat**: Where and how the mushroom grows
    
    ### ⚠️ Important Safety Notice:
    This application is for **educational purposes only**. Never consume wild mushrooms without proper expert identification. Misidentification can be fatal!
    """)

def main():
    # Header
    st.markdown('<h1 class="main-header">🍄 ShroomBoom</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict if a mushroom is edible or poisonous using machine learning</p>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["🏠 Home", "🔮 Model Prediction", "📊 Visualizations", "📈 Model Performance", "❓ Help"]
    )
    
    # Load data and models
    with st.spinner("Loading data and models..."):
        data = load_data()
        models, metadata = load_models_and_metadata()
    
    if data is None or models is None:
        st.error("Failed to load required files. Please check if all model files are present.")
        return
    
    # Main content based on navigation
    if page == "🏠 Home":
        show_home_page(data)
    elif page == "🔮 Model Prediction":
        show_prediction_page(models, metadata, data)
    elif page == "📊 Visualizations":
        show_visualization_page(data)
    elif page == "📈 Model Performance":
        show_performance_page(models, data)
    elif page == "❓ Help":
        show_help()

def show_home_page(data):
    """Display the home page with overview information"""
    st.markdown("## Welcome to ShroomBoom! 🍄")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Total Samples",
            value=f"{len(data):,}",
            help="Total number of mushroom samples in the dataset"
        )
    
    with col2:
        edible_count = len(data[data['class'] == 'e'])
        st.metric(
            label="Edible Mushrooms",
            value=f"{edible_count:,}",
            delta=f"{edible_count/len(data)*100:.1f}%"
        )
    
    with col3:
        poisonous_count = len(data[data['class'] == 'p'])
        st.metric(
            label="Poisonous Mushrooms",
            value=f"{poisonous_count:,}",
            delta=f"{poisonous_count/len(data)*100:.1f}%"
        )
    
    st.markdown("---")
    
    # Safety warning
    st.warning("""
    ⚠️ **IMPORTANT SAFETY WARNING** ⚠️
    
    This application is for **educational and research purposes only**. 
    
    **NEVER eat wild mushrooms** based solely on this prediction tool or any automated system. 
    Mushroom identification requires expert knowledge, and misidentification can result in serious illness or death.
    
    Always consult with mycology experts and use multiple reliable sources for mushroom identification.
    """)
    
    # Dataset overview
    st.markdown("## Dataset Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Class distribution pie chart
        class_counts = data['class'].value_counts()
        class_names = ['Edible' if x == 'e' else 'Poisonous' for x in class_counts.index]
        
        fig = px.pie(
            values=class_counts.values,
            names=class_names,
            title="Distribution of Edible vs Poisonous Mushrooms",
            color_discrete_map={'Edible': '#00d4aa', 'Poisonous': '#ff4b4b'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Quick Stats")
        st.info(f"**Features**: {len(data.columns)-1}")
        st.info(f"**Classes**: 2 (Edible/Poisonous)")
        st.info(f"**Data Quality**: No missing values")

def show_prediction_page(models, metadata, data):
    """Display the prediction interface"""
    st.markdown("## 🔮 Mushroom Classification Prediction")
    
    # Model selection
    selected_model = st.selectbox(
        "Choose a model:",
        list(models.keys()),
        help="Select which machine learning model to use for prediction"
    )
    
    st.markdown("---")
    
    # Feature input form
    with st.form("prediction_form"):
        st.markdown("### Enter Mushroom Characteristics:")
        
        feature_mappings = get_feature_mappings()
        user_inputs = {}
        
        # Create input fields in columns
        col1, col2, col3 = st.columns(3)
        
        features_to_exclude = ['veil-type', 'cap_combined']  # veil-type not in model, cap_combined created automatically
        available_features = [f for f in metadata['features'] if f not in features_to_exclude]
        
        for i, feature in enumerate(available_features):
            col = [col1, col2, col3][i % 3]
            
            with col:
                if feature in feature_mappings:
                    options = list(feature_mappings[feature].keys())
                    labels = [feature_mappings[feature][opt] for opt in options]
                    
                    selected_idx = st.selectbox(
                        f"{feature.replace('-', ' ').title()}:",
                        range(len(options)),
                        format_func=lambda x: labels[x],
                        key=feature
                    )
                    user_inputs[feature] = options[selected_idx]
                else:
                    # Fallback for features without mappings
                    unique_values = data[feature].unique() if feature in data.columns else ['unknown']
                    user_inputs[feature] = st.selectbox(
                        f"{feature.replace('-', ' ').title()}:",
                        unique_values,
                        key=feature
                    )
        
        # Predict button
        predict_button = st.form_submit_button(
            "🔮 Predict Mushroom Safety",
            use_container_width=True
        )
        
        if predict_button:
            with st.spinner("Making prediction..."):
                try:
                    # Prepare input data
                    input_df = pd.DataFrame([user_inputs])
                    
                    # Create cap_combined feature (combination of cap features)
                    if 'cap-shape' in input_df.columns and 'cap-surface' in input_df.columns and 'cap-color' in input_df.columns:
                        input_df['cap_combined'] = input_df['cap-shape'] + '_' + input_df['cap-surface'] + '_' + input_df['cap-color']
                    
                    # Make prediction
                    model = models[selected_model]
                    prediction_numeric = model.predict(input_df)[0]
                    # Convert numeric prediction back to string label
                    prediction = 'e' if prediction_numeric == 0 else 'p'
                    prediction_proba = model.predict_proba(input_df)[0]
                    
                    # Display results
                    st.markdown("---")
                    st.markdown("### 🎯 Prediction Results")
                    
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    with col2:
                        if prediction == 'e':
                            st.success("### ✅ EDIBLE")
                            st.markdown('<p class="safe">This mushroom is predicted to be SAFE to eat</p>', unsafe_allow_html=True)
                            confidence = prediction_proba[0] * 100  # Assuming class 0 is edible
                        else:
                            st.error("### ☠️ POISONOUS")
                            st.markdown('<p class="danger">This mushroom is predicted to be DANGEROUS</p>', unsafe_allow_html=True)
                            confidence = prediction_proba[1] * 100  # Assuming class 1 is poisonous
                        
                        st.metric("Confidence", f"{confidence:.1f}%")
                    
                    # Confidence breakdown
                    st.markdown("### Confidence Breakdown")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Edible Probability", f"{prediction_proba[0]*100:.1f}%")
                    with col2:
                        st.metric("Poisonous Probability", f"{prediction_proba[1]*100:.1f}%")
                    
                    # Visual confidence indicator
                    fig = go.Figure(go.Bar(
                        x=['Edible', 'Poisonous'],
                        y=[prediction_proba[0]*100, prediction_proba[1]*100],
                        marker_color=['#00d4aa', '#ff4b4b']
                    ))
                    fig.update_layout(
                        title="Prediction Confidence",
                        yaxis_title="Probability (%)",
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error making prediction: {e}")

def show_visualization_page(data):
    """Display data visualizations"""
    st.markdown("## 📊 Data Visualizations")
    
    # Feature distribution by class
    st.markdown("### Feature Distributions by Class")
    
    # Select feature to visualize
    categorical_features = [col for col in data.columns if col != 'class']
    selected_feature = st.selectbox(
        "Select a feature to visualize:",
        categorical_features
    )
    
    if selected_feature:
        # Create crosstab
        crosstab = pd.crosstab(data[selected_feature], data['class'])
        crosstab.columns = ['Edible', 'Poisonous']
        
        # Create stacked bar chart
        fig = px.bar(
            crosstab,
            title=f"Distribution of {selected_feature.replace('-', ' ').title()} by Class",
            color_discrete_map={'Edible': '#00d4aa', 'Poisonous': '#ff4b4b'}
        )
        fig.update_layout(
            xaxis_title=selected_feature.replace('-', ' ').title(),
            yaxis_title="Count"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap for encoded features
    st.markdown("### Feature Relationships")
    
    with st.spinner("Generating correlation heatmap..."):
        # Encode categorical variables for correlation analysis
        from sklearn.preprocessing import LabelEncoder
        
        data_encoded = data.copy()
        le = LabelEncoder()
        
        for col in data_encoded.columns:
            data_encoded[col] = le.fit_transform(data_encoded[col].astype(str))
        
        # Calculate correlation matrix
        corr_matrix = data_encoded.corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            corr_matrix, 
            annot=False, 
            cmap='coolwarm', 
            center=0,
            square=True,
            ax=ax
        )
        plt.title("Feature Correlation Matrix")
        plt.tight_layout()
        st.pyplot(fig)

def show_performance_page(models, data):
    """Display model performance metrics"""
    st.markdown("## 📈 Model Performance Analysis")
    
    # Performance metrics for each model
    st.markdown("### Model Comparison")
    
    # This is a simplified version - in practice, you'd want to evaluate on a test set
    performance_data = {
        'Model': list(models.keys()),
        'Accuracy': [],
        'Features Used': []
    }
    
    # Sample some data for quick evaluation
    sample_data = data.sample(min(1000, len(data)), random_state=42).copy()
    
    # Create the cap_combined feature for the sample data
    sample_data['cap_combined'] = (
        sample_data['cap-shape'].astype(str) + '_' + 
        sample_data['cap-surface'].astype(str) + '_' + 
        sample_data['cap-color'].astype(str)
    )
    
    X_sample = sample_data.drop('class', axis=1)
    y_sample = sample_data['class']
    
    # Remove veil-type if it exists (not used in models)
    if 'veil-type' in X_sample.columns:
        X_sample = X_sample.drop('veil-type', axis=1)
    
    for model_name, model in models.items():
        try:
            # Make predictions on sample
            y_pred_numeric = model.predict(X_sample)
            
            # Convert numeric predictions back to string labels
            # 0 = 'e' (edible), 1 = 'p' (poisonous)
            y_pred = pd.Series(['e' if pred == 0 else 'p' for pred in y_pred_numeric])
            y_sample_reset = y_sample.reset_index(drop=True)
            
            accuracy = (y_pred == y_sample_reset).mean()
            
            performance_data['Accuracy'].append(f"{accuracy*100:.2f}%")
            
            # Get number of features
            if hasattr(model, 'feature_names_in_'):
                features_count = len(model.feature_names_in_)
            elif hasattr(model, 'n_features_in_'):
                features_count = model.n_features_in_
            else:
                features_count = len(X_sample.columns)
            
            performance_data['Features Used'].append(features_count)
            
        except Exception as e:
            st.error(f"Error evaluating {model_name}: {str(e)}")
            performance_data['Accuracy'].append('Error')
            performance_data['Features Used'].append('Error')
    
    # Display performance table
    performance_df = pd.DataFrame(performance_data)
    st.dataframe(performance_df, use_container_width=True)
    
    # Model accuracy comparison chart
    valid_accuracies = []
    model_names = []
    
    for i, acc in enumerate(performance_data['Accuracy']):
        if acc != 'Error':
            valid_accuracies.append(float(acc.replace('%', '')))
            model_names.append(performance_data['Model'][i])
    
    if valid_accuracies:
        fig = px.bar(
            x=model_names,
            y=valid_accuracies,
            title="Model Accuracy Comparison",
            labels={'x': 'Model', 'y': 'Accuracy (%)'},
            color=valid_accuracies,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed Performance Metrics
    st.markdown("### Detailed Performance Metrics")
    
    # Create tabs for different metrics
    tab1, tab2, tab3 = st.tabs(["Classification Report", "Confusion Matrix", "Model Comparison"])
    
    with tab1:
        selected_model_report = st.selectbox(
            "Select model for classification report:",
            list(models.keys()),
            key="report_model"
        )
        
        if selected_model_report in models:
            try:
                model = models[selected_model_report]
                y_pred_numeric = model.predict(X_sample)
                # Convert numeric predictions back to string labels
                y_pred = ['e' if pred == 0 else 'p' for pred in y_pred_numeric]
                
                # Generate classification report
                from sklearn.metrics import classification_report, confusion_matrix
                
                report = classification_report(
                    y_sample, 
                    y_pred, 
                    target_names=['Edible', 'Poisonous'],
                    output_dict=True
                )
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Precision (Edible)", 
                        f"{report['Edible']['precision']:.3f}"
                    )
                    st.metric(
                        "Precision (Poisonous)", 
                        f"{report['Poisonous']['precision']:.3f}"
                    )
                
                with col2:
                    st.metric(
                        "Recall (Edible)", 
                        f"{report['Edible']['recall']:.3f}"
                    )
                    st.metric(
                        "Recall (Poisonous)", 
                        f"{report['Poisonous']['recall']:.3f}"
                    )
                
                with col3:
                    st.metric(
                        "F1-Score (Edible)", 
                        f"{report['Edible']['f1-score']:.3f}"
                    )
                    st.metric(
                        "F1-Score (Poisonous)", 
                        f"{report['Poisonous']['f1-score']:.3f}"
                    )
                
                # Overall metrics
                st.markdown("#### Overall Performance")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Overall Accuracy", f"{report['accuracy']:.3f}")
                with col2:
                    st.metric("Macro Average F1", f"{report['macro avg']['f1-score']:.3f}")
                with col3:
                    st.metric("Weighted Average F1", f"{report['weighted avg']['f1-score']:.3f}")
                
            except Exception as e:
                st.error(f"Error generating classification report: {str(e)}")
    
    with tab2:
        selected_model_cm = st.selectbox(
            "Select model for confusion matrix:",
            list(models.keys()),
            key="cm_model"
        )
        
        if selected_model_cm in models:
            try:
                model = models[selected_model_cm]
                y_pred_numeric = model.predict(X_sample)
                # Convert numeric predictions back to string labels
                y_pred = ['e' if pred == 0 else 'p' for pred in y_pred_numeric]
                
                # Create confusion matrix
                cm = confusion_matrix(y_sample, y_pred)
                
                # Create heatmap
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    xticklabels=['Edible', 'Poisonous'],
                    yticklabels=['Edible', 'Poisonous'],
                    ax=ax
                )
                ax.set_title(f'Confusion Matrix - {selected_model_cm}')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                
                st.pyplot(fig)
                
                # Calculate additional metrics
                tn, fp, fn, tp = cm.ravel()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("True Positives", tp)
                with col2:
                    st.metric("True Negatives", tn)
                with col3:
                    st.metric("False Positives", fp)
                with col4:
                    st.metric("False Negatives", fn)
                
            except Exception as e:
                st.error(f"Error generating confusion matrix: {str(e)}")
    
    with tab3:
        # Model comparison metrics
        if valid_accuracies:
            comparison_data = {
                'Model': model_names,
                'Accuracy (%)': valid_accuracies
            }
            
            # Add precision and recall for each model
            precisions = []
            recalls = []
            f1_scores = []
            
            for model_name in model_names:
                try:
                    model = models[model_name]
                    y_pred_numeric = model.predict(X_sample)
                    # Convert numeric predictions back to string labels
                    y_pred = ['e' if pred == 0 else 'p' for pred in y_pred_numeric]
                    
                    report = classification_report(
                        y_sample, 
                        y_pred,
                        output_dict=True,
                        zero_division=0
                    )
                    
                    # Use weighted averages
                    precisions.append(report['weighted avg']['precision'] * 100)
                    recalls.append(report['weighted avg']['recall'] * 100)
                    f1_scores.append(report['weighted avg']['f1-score'] * 100)
                    
                except:
                    precisions.append(0)
                    recalls.append(0)
                    f1_scores.append(0)
            
            comparison_data['Precision (%)'] = precisions
            comparison_data['Recall (%)'] = recalls
            comparison_data['F1-Score (%)'] = f1_scores
            
            # Create comparison chart
            df_comparison = pd.DataFrame(comparison_data)
            
            # Melt the dataframe for plotting
            df_melted = df_comparison.melt(
                id_vars=['Model'],
                value_vars=['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)'],
                var_name='Metric',
                value_name='Score'
            )
            
            fig = px.bar(
                df_melted,
                x='Model',
                y='Score',
                color='Metric',
                barmode='group',
                title='Model Performance Comparison',
                labels={'Score': 'Score (%)'}
            )
            
            fig.update_layout(yaxis=dict(range=[0, 100]))
            st.plotly_chart(fig, use_container_width=True)
            
            # Display the comparison table
            st.markdown("#### Performance Summary Table")
            st.dataframe(df_comparison, use_container_width=True)
    
    # Feature importance (if available)
    st.markdown("### Feature Importance Analysis")
    
    selected_model_perf = st.selectbox(
        "Select model for detailed analysis:",
        list(models.keys()),
        key="performance_model"
    )
    
    model = models[selected_model_perf]
    
    # Try to get feature importance
    if hasattr(model.named_steps.get('classifier', model), 'feature_importances_'):
        importance = model.named_steps.get('classifier', model).feature_importances_
        feature_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else [f'Feature_{i}' for i in range(len(importance))]
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(
            importance_df.tail(10),  # Show top 10
            x='Importance',
            y='Feature',
            orientation='h',
            title=f"Top 10 Feature Importances - {selected_model_perf}"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance not available for this model type.")

if __name__ == "__main__":
    main()