import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
from scipy.stats import zscore
import warnings
import zipfile
import io
warnings.filterwarnings('ignore')

st.set_page_config(page_title="US Real Estate Analysis", layout="wide", initial_sidebar_state="expanded")

@st.cache_data
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

@st.cache_data
def preprocess_data(df):
    df_clean = df.copy()
    
    # Remove extreme outliers using IQR method
    numeric_cols = ['price', 'bed', 'bath', 'acre_lot', 'house_size']
    for col in numeric_cols:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.01)
            Q3 = df_clean[col].quantile(0.99)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    # Remove rows with missing critical values
    df_clean = df_clean.dropna(subset=['price', 'bed', 'bath', 'state'])
    
    # Fill missing values
    df_clean['acre_lot'] = df_clean['acre_lot'].fillna(df_clean['acre_lot'].median())
    df_clean['house_size'] = df_clean['house_size'].fillna(df_clean['house_size'].median())
    
    # Remove invalid values
    df_clean = df_clean[df_clean['price'] > 0]
    df_clean = df_clean[df_clean['bed'] > 0]
    df_clean = df_clean[df_clean['bath'] > 0]
    
    return df_clean

@st.cache_data
def create_features(df):
    df_feat = df.copy()
    
    # Price per square foot
    df_feat['price_per_sqft'] = df_feat['price'] / (df_feat['house_size'] + 1)
    
    # Bath to bed ratio
    df_feat['bath_bed_ratio'] = df_feat['bath'] / (df_feat['bed'] + 1)
    
    # Total rooms
    df_feat['total_rooms'] = df_feat['bed'] + df_feat['bath']
    
    # Property size category
    df_feat['size_category'] = pd.cut(df_feat['house_size'], 
                                       bins=[0, 1000, 2000, 3000, 5000, 100000],
                                       labels=['Small', 'Medium', 'Large', 'Very Large', 'Mansion'])
    
    # Price category
    df_feat['price_category'] = pd.cut(df_feat['price'],
                                        bins=[0, 100000, 250000, 500000, 1000000, 10000000000],
                                        labels=['Budget', 'Affordable', 'Premium', 'Luxury', 'Ultra Luxury'])
    
    # Lot size category
    df_feat['lot_category'] = pd.cut(df_feat['acre_lot'],
                                      bins=[0, 0.25, 0.5, 1, 5, 10000],
                                      labels=['Small Lot', 'Medium Lot', 'Large Lot', 'Very Large', 'Estate'])
    
    # Status encoding
    status_map = {'for_sale': 1, 'sold': 0, 'ready_to_build': 2}
    df_feat['status_encoded'] = df_feat['status'].map(status_map).fillna(0)
    
    return df_feat

@st.cache_resource
def train_models(X_train, X_test, y_train, y_test):
    models = {}
    predictions = {}
    metrics = {}
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models['Linear Regression'] = lr
    predictions['Linear Regression'] = lr.predict(X_test)
    
    # Ridge Regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    models['Ridge'] = ridge
    predictions['Ridge'] = ridge.predict(X_test)
    
    # Lasso Regression
    lasso = Lasso(alpha=1.0)
    lasso.fit(X_train, y_train)
    models['Lasso'] = lasso
    predictions['Lasso'] = lasso.predict(X_test)
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    predictions['Random Forest'] = rf.predict(X_test)
    
    # Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    gb.fit(X_train, y_train)
    models['Gradient Boosting'] = gb
    predictions['Gradient Boosting'] = gb.predict(X_test)
    
    # XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=7, learning_rate=0.1, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    models['XGBoost'] = xgb_model
    predictions['XGBoost'] = xgb_model.predict(X_test)
    
    # LightGBM
    lgb_model = lgb.LGBMRegressor(n_estimators=100, max_depth=7, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)
    lgb_model.fit(X_train, y_train)
    models['LightGBM'] = lgb_model
    predictions['LightGBM'] = lgb_model.predict(X_test)
    
    # Calculate metrics
    for name, pred in predictions.items():
        r2 = r2_score(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        mae = mean_absolute_error(y_test, pred)
        mape = np.mean(np.abs((y_test - pred) / y_test)) * 100
        
        metrics[name] = {
            'R2 Score': r2,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }
    
    return models, predictions, metrics

def main():
    st.title("US Real Estate Market Analysis & Price Prediction")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "Executive Summary",
        "Exploratory Data Analysis",
        "Geographic Analysis",
        "Price Prediction Engine",
        "Market Segmentation",
        "Model Performance",
        "Feature Importance",
        "Bias & Fairness Analysis",
        "Market Volatility"
    ])
    
    # Load data
st.sidebar.header("Data")

uploaded = st.sidebar.file_uploader("Upload dataset (.csv or .zip)", type=["csv", "zip"])


# Use df_raw only if it loaded
df_raw = None
if uploaded is not None:
    try:
        if uploaded.name.endswith(".zip"):
            import zipfile
            with zipfile.ZipFile(uploaded) as z:
                csv_names = [n for n in z.namelist() if n.lower().endswith(".csv")]
                if not csv_names:
                    st.sidebar.error("No CSV file found inside the ZIP.")
                    st.stop()
                with z.open(csv_names[0]) as f:
                    df_raw = pd.read_csv(f)
        else:
            df_raw = pd.read_csv(uploaded)

        st.session_state["df_raw"] = df_raw
        df_clean = preprocess_data(df_raw)
        st.session_state["df_clean"] = df_clean
        df_feat = create_features(df_clean)
        st.session_state["df_feat"] = df_feat

        st.sidebar.success(f"Data loaded successfully! {len(df_clean):,} records")
    except Exception as e:
        st.sidebar.error(f"Error loading data: {e}")
        st.stop()

if "df_feat" not in st.session_state:
    st.info("Please upload the dataset using the sidebar.")
    st.stop()

df = st.session_state["df_feat"]
  
    # PAGE 1: EXECUTIVE SUMMARY
    if page == "Executive Summary":
        st.header("Executive Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Listings", f"{len(df):,}")
            st.metric("Average Price", f"${df['price'].mean():,.0f}")
        
        with col2:
            st.metric("Median Price", f"${df['price'].median():,.0f}")
            st.metric("Price Std Dev", f"${df['price'].std():,.0f}")
        
        with col3:
            st.metric("Total States", df['state'].nunique())
            st.metric("Total Cities", df['city'].nunique())
        
        with col4:
            st.metric("Avg Bedrooms", f"{df['bed'].mean():.1f}")
            st.metric("Avg Bathrooms", f"{df['bath'].mean():.1f}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Property Status Distribution")
            status_counts = df['status'].value_counts()
            fig = px.pie(values=status_counts.values, names=status_counts.index, 
                        title="Property Status Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Top 10 States by Average Price")
            state_prices = df.groupby('state')['price'].mean().sort_values(ascending=False).head(10)
            fig = px.bar(x=state_prices.values, y=state_prices.index, orientation='h',
                        labels={'x': 'Average Price', 'y': 'State'},
                        title="Top 10 Most Expensive States")
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Price Distribution by Category")
            fig = px.histogram(df, x='price_category', 
                             title="Properties by Price Category",
                             color='price_category')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("House Size Distribution")
            fig = px.histogram(df, x='size_category',
                             title="Properties by Size Category",
                             color='size_category')
            st.plotly_chart(fig, use_container_width=True)
    
    # PAGE 2: EXPLORATORY DATA ANALYSIS
    elif page == "Exploratory Data Analysis":
        st.header("Exploratory Data Analysis")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Distributions", "Correlations", "Relationships", "Statistics"])
        
        with tab1:
            st.subheader("Feature Distributions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(df, x='price', nbins=50, title="Price Distribution",
                                  labels={'price': 'Price ($)'})
                st.plotly_chart(fig, use_container_width=True)
                
                fig = px.histogram(df, x='bed', title="Bedroom Distribution",
                                  labels={'bed': 'Number of Bedrooms'})
                st.plotly_chart(fig, use_container_width=True)
                
                fig = px.histogram(df, x='acre_lot', nbins=50, title="Lot Size Distribution",
                                  labels={'acre_lot': 'Lot Size (acres)'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.histogram(df, x='house_size', nbins=50, title="House Size Distribution",
                                  labels={'house_size': 'House Size (sqft)'})
                st.plotly_chart(fig, use_container_width=True)
                
                fig = px.histogram(df, x='bath', title="Bathroom Distribution",
                                  labels={'bath': 'Number of Bathrooms'})
                st.plotly_chart(fig, use_container_width=True)
                
                fig = px.histogram(df, x='price_per_sqft', nbins=50, title="Price per Sqft Distribution",
                                  labels={'price_per_sqft': 'Price per Sqft ($)'})
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Correlation Analysis")
            
            numeric_cols = ['price', 'bed', 'bath', 'acre_lot', 'house_size', 'price_per_sqft', 
                          'bath_bed_ratio', 'total_rooms']
            corr_matrix = df[numeric_cols].corr()
            
            fig = px.imshow(corr_matrix, 
                          text_auto='.2f',
                          aspect="auto",
                          color_continuous_scale='RdBu_r',
                          title="Feature Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Correlation with Price")
            price_corr = corr_matrix['price'].sort_values(ascending=False)[1:]
            fig = px.bar(x=price_corr.values, y=price_corr.index, orientation='h',
                        title="Features Correlated with Price",
                        labels={'x': 'Correlation', 'y': 'Feature'})
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Feature Relationships")
            
            col1, col2 = st.columns(2)
            
            with col1:
                sample_df = df.sample(min(10000, len(df)))
                fig = px.scatter(sample_df, x='house_size', y='price', 
                               color='bed', size='bath',
                               title="Price vs House Size (colored by bedrooms)",
                               labels={'house_size': 'House Size (sqft)', 'price': 'Price ($)'})
                st.plotly_chart(fig, use_container_width=True)
                
                fig = px.box(df, x='bed', y='price', 
                           title="Price Distribution by Number of Bedrooms",
                           labels={'bed': 'Bedrooms', 'price': 'Price ($)'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.scatter(sample_df, x='acre_lot', y='price',
                               color='state',
                               title="Price vs Lot Size by State",
                               labels={'acre_lot': 'Lot Size (acres)', 'price': 'Price ($)'})
                st.plotly_chart(fig, use_container_width=True)
                
                fig = px.box(df, x='bath', y='price',
                           title="Price Distribution by Number of Bathrooms",
                           labels={'bath': 'Bathrooms', 'price': 'Price ($)'})
                st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("Statistical Summary")
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
            
            st.subheader("Top 20 Cities by Number of Listings")
            city_counts = df['city'].value_counts().head(20)
            fig = px.bar(x=city_counts.values, y=city_counts.index, orientation='h',
                        title="Top 20 Cities by Listings",
                        labels={'x': 'Number of Listings', 'y': 'City'})
            st.plotly_chart(fig, use_container_width=True)
    
    # PAGE 3: GEOGRAPHIC ANALYSIS
    elif page == "Geographic Analysis":
        st.header("Geographic Analysis")
        
        tab1, tab2, tab3 = st.tabs(["State Analysis", "City Analysis", "Regional Patterns"])
        
        with tab1:
            st.subheader("State-Level Analysis")
            
            state_stats = df.groupby('state').agg({
                'price': ['mean', 'median', 'std', 'count'],
                'house_size': 'mean',
                'bed': 'mean',
                'bath': 'mean'
            }).round(2)
            state_stats.columns = ['Avg Price', 'Median Price', 'Price Std', 'Count', 
                                  'Avg Size', 'Avg Beds', 'Avg Baths']
            state_stats = state_stats.sort_values('Avg Price', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Top 15 States by Average Price")
                top_states = state_stats.head(15)
                fig = px.bar(top_states, x=top_states.index, y='Avg Price',
                           title="Top 15 Most Expensive States",
                           labels={'x': 'State', 'Avg Price': 'Average Price ($)'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Top 15 States by Number of Listings")
                top_listings = state_stats.nlargest(15, 'Count')
                fig = px.bar(top_listings, x=top_listings.index, y='Count',
                           title="Top 15 States by Listings",
                           labels={'x': 'State', 'Count': 'Number of Listings'})
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("State Statistics Table")
            st.dataframe(state_stats, use_container_width=True)
            
            st.subheader("Price Variability by State")
            fig = px.bar(state_stats.head(20), x=state_stats.head(20).index, y='Price Std',
                        title="Top 20 States by Price Variability (Std Dev)",
                        labels={'x': 'State', 'Price Std': 'Price Standard Deviation ($)'})
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("City-Level Analysis")
            
            city_stats = df.groupby(['city', 'state']).agg({
                'price': ['mean', 'median', 'count']
            }).round(2)
            city_stats.columns = ['Avg Price', 'Median Price', 'Count']
            city_stats = city_stats[city_stats['Count'] >= 10]
            city_stats = city_stats.sort_values('Avg Price', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Top 20 Most Expensive Cities")
                top_cities = city_stats.head(20)
                fig = px.bar(top_cities, x=top_cities.index.get_level_values(0), 
                           y='Avg Price',
                           title="Top 20 Most Expensive Cities",
                           labels={'x': 'City', 'Avg Price': 'Average Price ($)'})
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Top 20 Most Affordable Cities")
                bottom_cities = city_stats.tail(20).sort_values('Avg Price', ascending=True)
                fig = px.bar(bottom_cities, x=bottom_cities.index.get_level_values(0),
                           y='Avg Price',
                           title="Top 20 Most Affordable Cities",
                           labels={'x': 'City', 'Avg Price': 'Average Price ($)'})
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Regional Price Patterns")
            
            col1, col2 = st.columns(2)
            
            with col1:
                state_price_size = df.groupby('state').agg({
                    'price': 'mean',
                    'house_size': 'mean'
                }).reset_index()
                
                fig = px.scatter(state_price_size, x='house_size', y='price',
                               text='state', size='price',
                               title="Average Price vs House Size by State",
                               labels={'house_size': 'Avg House Size (sqft)', 
                                      'price': 'Avg Price ($)'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                zip_stats = df.groupby('zip_code').agg({
                    'price': ['mean', 'count']
                }).reset_index()
                zip_stats.columns = ['zip_code', 'avg_price', 'count']
                zip_stats = zip_stats[zip_stats['count'] >= 5]
                zip_stats = zip_stats.sort_values('avg_price', ascending=False)
                
                fig = px.histogram(zip_stats, x='avg_price', nbins=50,
                                 title="Distribution of Average Prices by Zip Code",
                                 labels={'avg_price': 'Average Price ($)'})
                st.plotly_chart(fig, use_container_width=True)
    
    # PAGE 4: PRICE PREDICTION ENGINE
    elif page == "Price Prediction Engine":
        st.header("Price Prediction Engine")
        
        if 'models' not in st.session_state:
            with st.spinner("Training machine learning models..."):
                # Prepare data for modeling
                feature_cols = ['bed', 'bath', 'acre_lot', 'house_size', 'status_encoded']
                
                # Encode state as it's important
                state_encoder = LabelEncoder()
                df['state_encoded'] = state_encoder.fit_transform(df['state'])
                feature_cols.append('state_encoded')
                
                X = df[feature_cols].copy()
                y = df['price'].copy()
                
                # Remove any remaining NaN
                mask = ~(X.isna().any(axis=1) | y.isna())
                X = X[mask]
                y = y[mask]
                
                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train models
                models, predictions, metrics = train_models(X_train_scaled, X_test_scaled, 
                                                           y_train, y_test)
                
                st.session_state['models'] = models
                st.session_state['predictions'] = predictions
                st.session_state['metrics'] = metrics
                st.session_state['scaler'] = scaler
                st.session_state['state_encoder'] = state_encoder
                st.session_state['feature_cols'] = feature_cols
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
                
                st.success("Models trained successfully!")
        
        tab1, tab2 = st.tabs(["Make Prediction", "Prediction Analysis"])
        
        with tab1:
            st.subheader("Enter Property Details")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                beds = st.number_input("Number of Bedrooms", min_value=1, max_value=20, value=3)
                baths = st.number_input("Number of Bathrooms", min_value=1, max_value=20, value=2)
            
            with col2:
                house_size = st.number_input("House Size (sqft)", min_value=100, max_value=50000, value=2000)
                acre_lot = st.number_input("Lot Size (acres)", min_value=0.01, max_value=100.0, value=0.25, step=0.01)
            
            with col3:
                state = st.selectbox("State", sorted(df['state'].unique()))
                status = st.selectbox("Status", ['for_sale', 'sold', 'ready_to_build'])
            
            if st.button("Predict Price"):
                status_map = {'for_sale': 1, 'sold': 0, 'ready_to_build': 2}
                status_encoded = status_map[status]
                state_encoded = st.session_state['state_encoder'].transform([state])[0]
                
                input_data = pd.DataFrame({
                    'bed': [beds],
                    'bath': [baths],
                    'acre_lot': [acre_lot],
                    'house_size': [house_size],
                    'status_encoded': [status_encoded],
                    'state_encoded': [state_encoded]
                })
                
                input_scaled = st.session_state['scaler'].transform(input_data)
                
                st.subheader("Price Predictions by Model")
                
                predictions_dict = {}
                for name, model in st.session_state['models'].items():
                    pred = model.predict(input_scaled)[0]
                    predictions_dict[name] = pred
                
                pred_df = pd.DataFrame({
                    'Model': list(predictions_dict.keys()),
                    'Predicted Price': list(predictions_dict.values())
                })
                pred_df = pred_df.sort_values('Predicted Price', ascending=False)
                
                fig = px.bar(pred_df, x='Model', y='Predicted Price',
                           title="Price Predictions Across Models",
                           labels={'Predicted Price': 'Predicted Price ($)'})
                st.plotly_chart(fig, use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Prediction", f"${np.mean(list(predictions_dict.values())):,.0f}")
                with col2:
                    st.metric("Minimum Prediction", f"${np.min(list(predictions_dict.values())):,.0f}")
                with col3:
                    st.metric("Maximum Prediction", f"${np.max(list(predictions_dict.values())):,.0f}")
                
                st.dataframe(pred_df.style.format({'Predicted Price': '${:,.0f}'}), 
                           use_container_width=True)
        
        with tab2:
            st.subheader("Prediction vs Actual Analysis")
            
            best_model_name = max(st.session_state['metrics'].items(), 
                                 key=lambda x: x[1]['R2 Score'])[0]
            best_predictions = st.session_state['predictions'][best_model_name]
            
            sample_size = min(5000, len(st.session_state['y_test']))
            sample_indices = np.random.choice(len(st.session_state['y_test']), sample_size, replace=False)
            
            y_test_sample = st.session_state['y_test'].iloc[sample_indices]
            pred_sample = best_predictions[sample_indices]
            
            fig = px.scatter(x=y_test_sample, y=pred_sample,
                           labels={'x': 'Actual Price ($)', 'y': 'Predicted Price ($)'},
                           title=f"Actual vs Predicted Prices ({best_model_name})")
            
            min_val = min(y_test_sample.min(), pred_sample.min())
            max_val = max(y_test_sample.max(), pred_sample.max())
            fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                    mode='lines', name='Perfect Prediction',
                                    line=dict(color='red', dash='dash')))
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                errors = y_test_sample - pred_sample
                fig = px.histogram(errors, nbins=50,
                                 title="Prediction Error Distribution",
                                 labels={'value': 'Prediction Error ($)'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                residuals = (y_test_sample - pred_sample) / y_test_sample * 100
                fig = px.histogram(residuals, nbins=50,
                                 title="Percentage Error Distribution",
                                 labels={'value': 'Percentage Error (%)'})
                st.plotly_chart(fig, use_container_width=True)
    
    # PAGE 5: MARKET SEGMENTATION
    elif page == "Market Segmentation":
        st.header("Market Segmentation Analysis")
        
        if 'clusters' not in st.session_state:
            with st.spinner("Performing market segmentation..."):
                # Prepare data for clustering
                cluster_features = ['price', 'bed', 'bath', 'house_size', 'acre_lot', 'price_per_sqft']
                cluster_data = df[cluster_features].copy()
                
                # Remove outliers and NaN
                cluster_data = cluster_data.dropna()
                cluster_data = cluster_data[(np.abs(zscore(cluster_data)) < 3).all(axis=1)]
                
                # Scale data
                scaler = StandardScaler()
                cluster_scaled = scaler.fit_transform(cluster_data)
                
                # K-Means clustering
                kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(cluster_scaled)
                
                # PCA for visualization
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(cluster_scaled)
                
                cluster_data['Cluster'] = clusters
                cluster_data['PCA1'] = pca_result[:, 0]
                cluster_data['PCA2'] = pca_result[:, 1]
                
                st.session_state['cluster_data'] = cluster_data
                st.session_state['pca_variance'] = pca.explained_variance_ratio_
                
                st.success("Market segmentation completed!")
        
        cluster_data = st.session_state['cluster_data']
        
        tab1, tab2, tab3 = st.tabs(["Cluster Visualization", "Cluster Profiles", "Segment Analysis"])
        
        with tab1:
            st.subheader("Market Segments Visualization (PCA)")
            
            sample_size = min(10000, len(cluster_data))
            sample_data = cluster_data.sample(sample_size)
            
            fig = px.scatter(sample_data, x='PCA1', y='PCA2', color='Cluster',
                           title="Market Segments (PCA Reduced)",
                           labels={'PCA1': f'PC1 ({st.session_state["pca_variance"][0]:.1%} var)',
                                  'PCA2': f'PC2 ({st.session_state["pca_variance"][1]:.1%} var)'})
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.scatter(sample_data, x='price', y='house_size', color='Cluster',
                               title="Clusters: Price vs House Size",
                               labels={'price': 'Price ($)', 'house_size': 'House Size (sqft)'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.scatter(sample_data, x='bed', y='bath', color='Cluster',
                               title="Clusters: Bedrooms vs Bathrooms",
                               labels={'bed': 'Bedrooms', 'bath': 'Bathrooms'})
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Cluster Profiles")
            
            cluster_profiles = cluster_data.groupby('Cluster').agg({
                'price': ['mean', 'median', 'std', 'count'],
                'bed': 'mean',
                'bath': 'mean',
                'house_size': 'mean',
                'acre_lot': 'mean',
                'price_per_sqft': 'mean'
            }).round(2)
            
            cluster_profiles.columns = ['Avg Price', 'Median Price', 'Price Std', 'Count',
                                       'Avg Beds', 'Avg Baths', 'Avg Size', 'Avg Lot', 'Avg Price/Sqft']
            
            st.dataframe(cluster_profiles.style.format({
                'Avg Price': '${:,.0f}',
                'Median Price': '${:,.0f}',
                'Price Std': '${:,.0f}',
                'Avg Price/Sqft': '${:,.2f}'
            }), use_container_width=True)
            
            cluster_sizes = cluster_data['Cluster'].value_counts().sort_index()
            fig = px.pie(values=cluster_sizes.values, names=cluster_sizes.index,
                        title="Market Segment Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Detailed Segment Analysis")
            
            selected_cluster = st.selectbox("Select Segment", sorted(cluster_data['Cluster'].unique()))
            
            cluster_subset = cluster_data[cluster_data['Cluster'] == selected_cluster]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Properties", f"{len(cluster_subset):,}")
            with col2:
                st.metric("Avg Price", f"${cluster_subset['price'].mean():,.0f}")
            with col3:
                st.metric("Avg Size", f"{cluster_subset['house_size'].mean():,.0f} sqft")
            with col4:
                st.metric("Price/Sqft", f"${cluster_subset['price_per_sqft'].mean():.2f}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(cluster_subset, x='price', nbins=50,
                                 title=f"Price Distribution - Segment {selected_cluster}")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.histogram(cluster_subset, x='house_size', nbins=50,
                                 title=f"Size Distribution - Segment {selected_cluster}")
                st.plotly_chart(fig, use_container_width=True)
    
    # PAGE 6: MODEL PERFORMANCE
    elif page == "Model Performance":
        st.header("Model Performance Comparison")
        
        if 'metrics' not in st.session_state:
            st.warning("Please train models first by visiting the Price Prediction Engine page.")
            st.stop
        
        metrics_df = pd.DataFrame(st.session_state['metrics']).T
        
        st.subheader("Performance Metrics Comparison")
        st.dataframe(metrics_df.style.format({
            'R2 Score': '{:.4f}',
            'RMSE': '{:,.0f}',
            'MAE': '{:,.0f}',
            'MAPE': '{:.2f}%'
        }), use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(metrics_df.reset_index(), x='index', y='R2 Score',
                        title="R² Score Comparison (Higher is Better)",
                        labels={'index': 'Model', 'R2 Score': 'R² Score'})
            st.plotly_chart(fig, use_container_width=True)
            
            fig = px.bar(metrics_df.reset_index(), x='index', y='MAE',
                        title="Mean Absolute Error (Lower is Better)",
                        labels={'index': 'Model', 'MAE': 'MAE ($)'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(metrics_df.reset_index(), x='index', y='RMSE',
                        title="Root Mean Squared Error (Lower is Better)",
                        labels={'index': 'Model', 'RMSE': 'RMSE ($)'})
            st.plotly_chart(fig, use_container_width=True)
            
            fig = px.bar(metrics_df.reset_index(), x='index', y='MAPE',
                        title="Mean Absolute Percentage Error (Lower is Better)",
                        labels={'index': 'Model', 'MAPE': 'MAPE (%)'})
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Best Model")
        best_model = metrics_df['R2 Score'].idxmax()
        st.success(f"Best performing model: {best_model} with R² = {metrics_df.loc[best_model, 'R2 Score']:.4f}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R² Score", f"{metrics_df.loc[best_model, 'R2 Score']:.4f}")
        with col2:
            st.metric("RMSE", f"${metrics_df.loc[best_model, 'RMSE']:,.0f}")
        with col3:
            st.metric("MAPE", f"{metrics_df.loc[best_model, 'MAPE']:.2f}%")
    
    # PAGE 7: FEATURE IMPORTANCE
    elif page == "Feature Importance":
        st.header("Feature Importance Analysis")
        
        if 'models' not in st.session_state:
            st.warning("Please train models first by visiting the Price Prediction Engine page.")
            st.stop
        
        feature_names = ['Bedrooms', 'Bathrooms', 'Lot Size', 'House Size', 'Status', 'State']
        
        tab1, tab2 = st.tabs(["Tree-Based Models", "Linear Models"])
        
        with tab1:
            st.subheader("Feature Importance from Tree-Based Models")
            
            tree_models = ['Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM']
            
            for model_name in tree_models:
                if model_name in st.session_state['models']:
                    model = st.session_state['models'][model_name]
                    
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                        
                        importance_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance': importances
                        }).sort_values('Importance', ascending=False)
                        
                        fig = px.bar(importance_df, x='Importance', y='Feature',
                                   orientation='h',
                                   title=f"Feature Importance - {model_name}")
                        st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Feature Coefficients from Linear Models")
            
            linear_models = ['Linear Regression', 'Ridge', 'Lasso']
            
            for model_name in linear_models:
                if model_name in st.session_state['models']:
                    model = st.session_state['models'][model_name]
                    
                    if hasattr(model, 'coef_'):
                        coef = model.coef_
                        
                        coef_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Coefficient': coef
                        }).sort_values('Coefficient', ascending=False)
                        
                        fig = px.bar(coef_df, x='Coefficient', y='Feature',
                                   orientation='h',
                                   title=f"Feature Coefficients - {model_name}")
                        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Overall Feature Impact Summary")
        
        all_importances = []
        for model_name in tree_models:
            if model_name in st.session_state['models']:
                model = st.session_state['models'][model_name]
                if hasattr(model, 'feature_importances_'):
                    all_importances.append(model.feature_importances_)
        
        if all_importances:
            avg_importance = np.mean(all_importances, axis=0)
            importance_summary = pd.DataFrame({
                'Feature': feature_names,
                'Average Importance': avg_importance
            }).sort_values('Average Importance', ascending=False)
            
            fig = px.bar(importance_summary, x='Average Importance', y='Feature',
                       orientation='h',
                       title="Average Feature Importance Across Tree Models")
            st.plotly_chart(fig, use_container_width=True)
    
    # PAGE 8: BIAS & FAIRNESS ANALYSIS
    elif page == "Bias & Fairness Analysis":
        st.header("Bias & Fairness Analysis")
        
        if 'models' not in st.session_state:
            st.warning("Please train models first by visiting the Price Prediction Engine page.")
            st.stop
        
        tab1, tab2, tab3 = st.tabs(["Geographic Bias", "Price Range Analysis", "Prediction Equity"])
        
        with tab1:
            st.subheader("Prediction Error by State")
            
            best_model_name = max(st.session_state['metrics'].items(),
                                 key=lambda x: x[1]['R2 Score'])[0]
            best_predictions = st.session_state['predictions'][best_model_name]
            
            X_test = st.session_state['X_test']
            y_test = st.session_state['y_test']
            
            test_df = X_test.copy()
            test_df['actual_price'] = y_test.values
            test_df['predicted_price'] = best_predictions
            test_df['error'] = test_df['actual_price'] - test_df['predicted_price']
            test_df['abs_error'] = np.abs(test_df['error'])
            test_df['pct_error'] = (test_df['error'] / test_df['actual_price']) * 100
            
            state_encoder = st.session_state['state_encoder']
            test_df['state'] = state_encoder.inverse_transform(test_df['state_encoded'].astype(int))
            
            state_errors = test_df.groupby('state').agg({
                'abs_error': 'mean',
                'pct_error': 'mean',
                'error': ['mean', 'std', 'count']
            }).round(2)
            state_errors.columns = ['Mean Abs Error', 'Mean Pct Error', 'Mean Error', 'Std Error', 'Count']
            state_errors = state_errors[state_errors['Count'] >= 10]
            state_errors = state_errors.sort_values('Mean Abs Error', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                top_errors = state_errors.head(15)
                fig = px.bar(top_errors, x=top_errors.index, y='Mean Abs Error',
                           title="Top 15 States by Prediction Error",
                           labels={'x': 'State', 'Mean Abs Error': 'Mean Absolute Error ($)'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(state_errors.head(15), x=state_errors.head(15).index, 
                           y='Mean Pct Error',
                           title="Top 15 States by Percentage Error",
                           labels={'x': 'State', 'Mean Pct Error': 'Mean Percentage Error (%)'})
                st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(state_errors.head(20), use_container_width=True)
        
        with tab2:
            st.subheader("Prediction Performance by Price Range")
            
            test_df['price_range'] = pd.cut(test_df['actual_price'],
                                           bins=[0, 100000, 250000, 500000, 1000000, 100000000],
                                           labels=['<100K', '100K-250K', '250K-500K', '500K-1M', '>1M'])
            
            price_range_errors = test_df.groupby('price_range').agg({
                'abs_error': 'mean',
                'pct_error': 'mean',
                'error': 'count'
            }).round(2)
            price_range_errors.columns = ['Mean Abs Error', 'Mean Pct Error', 'Count']
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(price_range_errors, x=price_range_errors.index, y='Mean Abs Error',
                           title="Prediction Error by Price Range",
                           labels={'x': 'Price Range', 'Mean Abs Error': 'Mean Absolute Error ($)'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(price_range_errors, x=price_range_errors.index, y='Mean Pct Error',
                           title="Percentage Error by Price Range",
                           labels={'x': 'Price Range', 'Mean Pct Error': 'Mean Percentage Error (%)'})
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Over/Under Valuation Analysis")
            
            test_df['valuation'] = test_df['predicted_price'] - test_df['actual_price']
            test_df['valuation_pct'] = (test_df['valuation'] / test_df['actual_price']) * 100
            
            state_valuation = test_df.groupby('state')['valuation_pct'].mean().sort_values()
            state_valuation_df = state_valuation.reset_index()
            state_valuation_df.columns = ['State', 'Avg Valuation %']
            
            state_valuation_df['Valuation Type'] = state_valuation_df['Avg Valuation %'].apply(
                lambda x: 'Overvalued' if x > 0 else 'Undervalued'
            )
            
            fig = px.bar(state_valuation_df, x='State', y='Avg Valuation %',
                        color='Valuation Type',
                        title="Over/Under Valuation by State",
                        labels={'Avg Valuation %': 'Average Valuation (%)'},
                        color_discrete_map={'Overvalued': 'red', 'Undervalued': 'green'})
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Most Overvalued States")
                overvalued = state_valuation_df.nlargest(10, 'Avg Valuation %')
                st.dataframe(overvalued, use_container_width=True)
            
            with col2:
                st.subheader("Most Undervalued States")
                undervalued = state_valuation_df.nsmallest(10, 'Avg Valuation %')
                st.dataframe(undervalued, use_container_width=True)
    
    # PAGE 9: MARKET VOLATILITY
    elif page == "Market Volatility":
        st.header("Market Volatility Analysis")
        
        tab1, tab2, tab3 = st.tabs(["Price Volatility", "Regional Variation", "Risk Assessment"])
        
        with tab1:
            st.subheader("Price Volatility by State")
            
            state_volatility = df.groupby('state').agg({
                'price': ['mean', 'std', 'count']
            }).round(2)
            state_volatility.columns = ['Mean Price', 'Std Dev', 'Count']
            state_volatility['CoV'] = (state_volatility['Std Dev'] / state_volatility['Mean Price']) * 100
            state_volatility = state_volatility[state_volatility['Count'] >= 20]
            state_volatility = state_volatility.sort_values('CoV', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(state_volatility.head(20), x=state_volatility.head(20).index, y='CoV',
                           title="Top 20 States by Price Volatility (CoV)",
                           labels={'x': 'State', 'CoV': 'Coefficient of Variation (%)'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.scatter(state_volatility, x='Mean Price', y='Std Dev',
                               size='Count', hover_name=state_volatility.index,
                               title="Price Volatility vs Average Price",
                               labels={'Mean Price': 'Average Price ($)', 
                                      'Std Dev': 'Standard Deviation ($)'})
                st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(state_volatility.head(20), use_container_width=True)
        
        with tab2:
            st.subheader("City-Level Price Variation")
            
            city_volatility = df.groupby(['city', 'state']).agg({
                'price': ['mean', 'std', 'count']
            }).round(2)
            city_volatility.columns = ['Mean Price', 'Std Dev', 'Count']
            city_volatility['CoV'] = (city_volatility['Std Dev'] / city_volatility['Mean Price']) * 100
            city_volatility = city_volatility[city_volatility['Count'] >= 20]
            city_volatility = city_volatility.sort_values('CoV', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Most Volatile Cities")
                top_volatile = city_volatility.head(20)
                fig = px.bar(top_volatile, x=top_volatile.index.get_level_values(0), y='CoV',
                           title="Top 20 Most Volatile Cities",
                           labels={'x': 'City', 'CoV': 'Coefficient of Variation (%)'})
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Most Stable Cities")
                bottom_volatile = city_volatility.tail(20).sort_values('CoV', ascending=True)
                fig = px.bar(bottom_volatile, x=bottom_volatile.index.get_level_values(0), y='CoV',
                           title="Top 20 Most Stable Cities",
                           labels={'x': 'City', 'CoV': 'Coefficient of Variation (%)'})
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Market Risk Assessment")
            
            state_risk = df.groupby('state').agg({
                'price': ['mean', 'std', 'min', 'max', 'count']
            }).round(2)
            state_risk.columns = ['Mean', 'Std', 'Min', 'Max', 'Count']
            state_risk['Range'] = state_risk['Max'] - state_risk['Min']
            state_risk['CoV'] = (state_risk['Std'] / state_risk['Mean']) * 100
            state_risk = state_risk[state_risk['Count'] >= 20]
            
            state_risk['Risk Score'] = (
                (state_risk['CoV'] / state_risk['CoV'].max() * 0.5) +
                (state_risk['Range'] / state_risk['Range'].max() * 0.5)
            ) * 100
            
            state_risk = state_risk.sort_values('Risk Score', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("High Risk Markets")
                high_risk = state_risk.head(15)
                fig = px.bar(high_risk, x=high_risk.index, y='Risk Score',
                           title="Top 15 High Risk Markets",
                           labels={'x': 'State', 'Risk Score': 'Risk Score'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Low Risk Markets")
                low_risk = state_risk.tail(15).sort_values('Risk Score', ascending=True)
                fig = px.bar(low_risk, x=low_risk.index, y='Risk Score',
                           title="Top 15 Low Risk Markets",
                           labels={'x': 'State', 'Risk Score': 'Risk Score'})
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Risk Metrics by State")
            st.dataframe(state_risk.head(20), use_container_width=True)

if __name__ == "__main__":
    main()
