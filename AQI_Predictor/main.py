import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
import xgboost as xgb

# ====================== GLOBAL AQI FUNCTIONS (DEFINED ONCE) ======================
def calculate_aqi(Cp, Ih, Il, BPh, BPl):
    return round(((Ih - Il) / (BPh - BPl)) * (Cp - BPl) + Il)

def aqi_pm25(conc):
    conc = float(conc)
    if 0 <= conc <= 12.0:
        return calculate_aqi(conc, 50, 0, 12.0, 0)
    elif 12.1 <= conc <= 35.4:
        return calculate_aqi(conc, 100, 51, 35.4, 12.1)
    elif 35.5 <= conc <= 55.4:
        return calculate_aqi(conc, 150, 101, 55.4, 35.5)
    elif 55.5 <= conc <= 150.4:
        return calculate_aqi(conc, 200, 151, 150.4, 55.5)
    elif 150.5 <= conc <= 250.4:
        return calculate_aqi(conc, 300, 201, 250.4, 150.5)
    elif 250.5 <= conc <= 350.4:
        return calculate_aqi(conc, 400, 301, 350.4, 250.5)
    elif 350.5 <= conc <= 500.4:
        return calculate_aqi(conc, 500, 401, 500.4, 350.5)
    elif conc > 500.4:
        return 500
    else:
        return None

def aqi_pm10(conc):
    conc = float(conc)
    if 0 <= conc <= 54:
        return calculate_aqi(conc, 50, 0, 54, 0)
    elif 55 <= conc <= 154:
        return calculate_aqi(conc, 100, 51, 154, 55)
    elif 155 <= conc <= 254:
        return calculate_aqi(conc, 150, 101, 254, 155)
    elif 255 <= conc <= 354:
        return calculate_aqi(conc, 200, 151, 354, 255)
    elif 355 <= conc <= 424:
        return calculate_aqi(conc, 300, 201, 424, 355)
    elif 425 <= conc <= 504:
        return calculate_aqi(conc, 400, 301, 504, 425)
    elif 505 <= conc <= 604:
        return calculate_aqi(conc, 500, 401, 604, 505)
    elif conc > 604:
        return 500
    else:
        return None

def aqi_no2(conc):
    conc = float(conc)
    if 0 <= conc <= 53:
        return calculate_aqi(conc, 50, 0, 53, 0)
    elif 54 <= conc <= 100:
        return calculate_aqi(conc, 100, 51, 100, 54)
    elif 101 <= conc <= 360:
        return calculate_aqi(conc, 150, 101, 360, 101)
    elif 361 <= conc <= 649:
        return calculate_aqi(conc, 200, 151, 649, 361)
    elif 650 <= conc <= 1249:
        return calculate_aqi(conc, 300, 201, 1249, 650)
    elif 1250 <= conc <= 1649:
        return calculate_aqi(conc, 400, 301, 1649, 1250)
    elif 1650 <= conc <= 2049:
        return calculate_aqi(conc, 500, 401, 2049, 1650)
    elif conc > 2049:
        return 500
    else:
        return None

def aqi_so2(conc):
    conc = float(conc)
    if 0 <= conc <= 35:
        return calculate_aqi(conc, 50, 0, 35, 0)
    elif 36 <= conc <= 75:
        return calculate_aqi(conc, 100, 51, 75, 36)
    elif 76 <= conc <= 185:
        return calculate_aqi(conc, 150, 101, 185, 76)
    elif 186 <= conc <= 304:
        return calculate_aqi(conc, 200, 151, 304, 186)
    elif 305 <= conc <= 604:
        return calculate_aqi(conc, 300, 201, 604, 305)
    elif 605 <= conc <= 804:
        return calculate_aqi(conc, 400, 301, 804, 605)
    elif 805 <= conc <= 1004:
        return calculate_aqi(conc, 500, 401, 1004, 805)
    elif conc > 1004:
        return 500
    else:
        return None

def aqi_co(conc):
    conc_ppm = float(conc) * 0.87
    if 0 <= conc_ppm <= 4.4:
        return calculate_aqi(conc_ppm, 50, 0, 4.4, 0)
    elif 4.5 <= conc_ppm <= 9.4:
        return calculate_aqi(conc_ppm, 100, 51, 9.4, 4.5)
    elif 9.5 <= conc_ppm <= 12.4:
        return calculate_aqi(conc_ppm, 150, 101, 12.4, 9.5)
    elif 12.5 <= conc_ppm <= 15.4:
        return calculate_aqi(conc_ppm, 200, 151, 15.4, 12.5)
    elif 15.5 <= conc_ppm <= 30.4:
        return calculate_aqi(conc_ppm, 300, 201, 30.4, 15.5)
    elif 30.5 <= conc_ppm <= 40.4:
        return calculate_aqi(conc_ppm, 400, 301, 40.4, 30.5)
    elif 40.5 <= conc_ppm <= 50.4:
        return calculate_aqi(conc_ppm, 500, 401, 50.4, 40.5)
    elif conc_ppm > 50.4:
        return 500
    else:
        return None

def aqi_o3(conc):
    conc = float(conc)
    if 0 <= conc <= 124:
        return calculate_aqi(conc, 100, 0, 124, 0)
    elif 125 <= conc <= 164:
        return calculate_aqi(conc, 150, 101, 164, 125)
    elif 165 <= conc <= 204:
        return calculate_aqi(conc, 200, 151, 204, 165)
    elif 205 <= conc <= 404:
        return calculate_aqi(conc, 300, 201, 404, 205)
    elif 405 <= conc <= 504:
        return calculate_aqi(conc, 400, 301, 504, 405)
    elif 505 <= conc <= 604:
        return calculate_aqi(conc, 500, 401, 604, 505)
    elif conc > 604:
        return 500
    else:
        return None

def aqi_category(aqi):
    if aqi is None:
        return "Invalid"
    aqi = int(aqi)
    if 0 <= aqi <= 50:
        return "Good"
    elif 51 <= aqi <= 100:
        return "Moderate"
    elif 101 <= aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif 151 <= aqi <= 200:
        return "Unhealthy"
    elif 201 <= aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

# Page config
st.set_page_config(page_title="Global Air Quality Analysis Dashboard", layout="wide")
st.title("🌍 Global Air Quality Data Analysis & AQI Prediction Dashboard")

# Sidebar
page = st.sidebar.selectbox("Choose a Section", [
    "Data Upload & Overview",
    "Data Cleaning & Preprocessing",
    "Exploratory Data Analysis (EDA)",
    "AQI Calculation",
    "Regression Models for AQI Prediction",
    "Classification Models for AQI Category",
    "Feature Importance Summary",
    "Prediction Tool"
])

st.sidebar.markdown("### Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload `global_air_quality_data_10000.csv`", type=["csv"])

if uploaded_file is None:
    st.warning("Please upload the CSV file to proceed.")
    st.stop()

df_raw = pd.read_csv(uploaded_file)

# Common features
features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity', 'Wind Speed', 'Month', 'Day']

# ====================== Pages ======================
if page == "Data Upload & Overview":
    st.header("Data Overview")
    st.dataframe(df_raw.head())
    st.write("**Shape:**", df_raw.shape)
    st.write("**Columns:**", list(df_raw.columns))
    st.write("**Missing Values:**", df_raw.isnull().sum().sum())
    st.write("**Duplicates:**", df_raw.duplicated().sum())

elif page == "Data Cleaning & Preprocessing":
    st.header("Data Cleaning & Preprocessing")
    
    df = df_raw.copy()
    null_sum = df.isnull().sum().sum()
    dup_sum = df.duplicated().sum()
    st.write("**Total Null Values:**", null_sum)
    st.write("**Duplicates:**", dup_sum)
    
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Year'] = df['Date'].dt.year
    df = df.sort_values(by='Date', ascending=True)
    df.drop('Date', axis=1, inplace=True)
    df.drop('Year', axis=1, inplace=True)
    
    st.subheader("Skewness")
    cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity', 'Wind Speed']
    skew_df = pd.DataFrame({col: [df[col].skew()] for col in cols}).T
    skew_df.columns = ['Skewness']
    st.dataframe(skew_df)
    
    st.subheader("Outliers Detection (Z-score > 3)")
    for col in cols:
        zscores = stats.zscore(df[col])
        outliers = np.abs(zscores) > 3
        st.write(f"**{col} Outliers:**")
        st.dataframe(df.loc[outliers, col])
    
    st.write("**Cleaned Data Preview:**")
    st.dataframe(df.head())


elif page == "Exploratory Data Analysis (EDA)":
    st.header("Exploratory Data Analysis")
    
    df = df_raw.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df = df.sort_values(by='Date', ascending=True)
    df.drop('Date', axis=1, inplace=True)
    
    month_map = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
                 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
    df['Month_Name'] = df['Month'].map(month_map)
    
    columns_to_analyze = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity', 'Wind Speed']
    colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
              'rgb(148, 103, 189)', 'rgb(140, 86, 75)', 'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
              'rgb(188, 189, 34)']
    
    tab_dist, tab_counts, tab_avg, tab_corr, tab_season = st.tabs(["Distributions", "Counts", "Averages", "Correlations", "Seasonal Trends"])
    
    with tab_dist:
        st.subheader("Histograms")
        fig_hist = make_subplots(rows=3, cols=3, subplot_titles=columns_to_analyze)
        for i, (column, color) in enumerate(zip(columns_to_analyze, colors)):
            row = i // 3 + 1
            col = i % 3 + 1
            fig_hist.add_trace(go.Histogram(x=df[column], histnorm='density', marker_color=color), row=row, col=col)
        fig_hist.update_layout(height=900, width=1200, title_text="Histograms of Pollutants & Meteorological Features")
        st.plotly_chart(fig_hist, use_container_width=True)
        
        st.subheader("Box Plots")
        fig_box = make_subplots(rows=3, cols=3, subplot_titles=columns_to_analyze)
        for i, column in enumerate(columns_to_analyze):
            row = i // 3 + 1
            col = i % 3 + 1
            fig_box.add_trace(go.Box(y=df[column], boxmean=True, fillcolor=colors[i]), row=row, col=col)
        fig_box.update_layout(height=900, width=1200, title_text='Box Plots of Pollutants and Meteorological Data')
        st.plotly_chart(fig_box, use_container_width=True)
        
        st.subheader("Violin Plots")
        fig_violin = make_subplots(rows=3, cols=3, subplot_titles=columns_to_analyze)
        for i, (column, color) in enumerate(zip(columns_to_analyze, colors)):
            row = i // 3 + 1
            col = i % 3 + 1
            fig_violin.add_trace(go.Violin(y=df[column], box_visible=True, meanline_visible=True, fillcolor=color, opacity=0.7), row=row, col=col)
        fig_violin.update_layout(height=900, width=1200, title_text="Violin Plots of Pollutants & Meteorological Features")
        st.plotly_chart(fig_violin, use_container_width=True)
    
    with tab_counts:
        st.subheader("Records per Month")
        month_counts = df['Month_Name'].value_counts().reindex(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
        fig_month_bar = px.bar(x=month_counts.index, y=month_counts.values, title='Records per Month', color=month_counts.index)
        st.plotly_chart(fig_month_bar)
        
        fig_month_pie = px.pie(names=month_counts.index, values=month_counts.values, title='Distribution of Records by Month')
        st.plotly_chart(fig_month_pie)
        
        st.subheader("Records per City")
        city_counts = df['City'].value_counts()
        fig_city_bar = px.bar(x=city_counts.index, y=city_counts.values, title='Records per City', color=city_counts.index)
        st.plotly_chart(fig_city_bar)
        
        fig_city_pie = px.pie(names=city_counts.index, values=city_counts.values, title='Distribution of Records by City')
        st.plotly_chart(fig_city_pie)
        
        st.subheader("Records per Country")
        country_counts = df['Country'].value_counts()
        fig_country_bar = px.bar(x=country_counts.index, y=country_counts.values, title='Records per Country', color=country_counts.index)
        st.plotly_chart(fig_country_bar)
        
        fig_country_pie = px.pie(names=country_counts.index, values=country_counts.values, title='Distribution of Records by Country')
        st.plotly_chart(fig_country_pie)
    
    with tab_avg:
        st.subheader("Average Pollutant Concentration per Country")
        pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
        df_country = df.groupby('Country')[pollutants].mean().reset_index()
        fig_country_line = px.line(df_country, x='Country', y=pollutants, title='Average Pollutant Concentration per Country', markers=True)
        st.plotly_chart(fig_country_line)
        
        st.subheader("Average Pollutant Concentration per City")
        df_city = df.groupby('City')[pollutants].mean().reset_index()
        fig_city_line = px.line(df_city, x='City', y=pollutants, title='Average Pollutant Concentration per City', markers=True)
        st.plotly_chart(fig_city_line)
    
    with tab_corr:
        st.subheader("Correlation Heatmap")
        num_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity', 'Wind Speed']
        corr_matrix = df[num_cols].corr()
        st.write(corr_matrix)
        
        fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r', zmin=-1, zmax=1, title='Correlation Heatmap')
        fig_corr.update_layout(width=900, height=800)
        st.plotly_chart(fig_corr)
        
        st.subheader("Pairwise Correlations")
        for i in range(len(num_cols)-1):
            for j in range(i+1, len(num_cols)):
                col1 = num_cols[i]
                col2 = num_cols[j]
                corr = df[col1].corr(df[col2])
                st.write(f"Correlation between {col1} and {col2}: {corr:.2f}")
    
    with tab_season:
        st.subheader("Month-by-Month Changes")
        meteo_vars = ['Temperature', 'Humidity', 'Wind Speed']
        df_grouped = df.groupby(['Country', 'Month_Name'])[pollutants + meteo_vars].mean().reset_index()
        
        for var in pollutants + meteo_vars:
            df_var = df_grouped[['Country', 'Month_Name', var]].rename(columns={var: 'Average_Value'})
            fig_bar = px.bar(df_var, x='Month_Name', y='Average_Value', color='Country', barmode='group', title=f'Month-by-Month Change of {var}')
            fig_bar.update_xaxes(categoryorder='array', categoryarray=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
            st.plotly_chart(fig_bar)
            
            fig_line = px.line(df_var, x='Month_Name', y='Average_Value', color='Country', markers=True, title=f'Month-by-Month Line of {var}')
            fig_line.update_xaxes(categoryorder='array', categoryarray=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
            st.plotly_chart(fig_line)
        
        st.subheader("Seasonal Variation (Overall)")
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        df['Month_Name'] = pd.Categorical(df['Month_Name'], categories=month_order, ordered=True)
        all_features = pollutants + meteo_vars
        for feature in all_features:
            monthly_avg = df.groupby('Month_Name')[feature].mean().reindex(month_order)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(x=monthly_avg.index, y=monthly_avg.values, marker='o', ax=ax)
            ax.set_title(f'Seasonal Variation: Average {feature} by Month')
            ax.set_xlabel('Month')
            ax.set_ylabel(f'Average {feature}')
            ax.grid(True)
            st.pyplot(fig)
        
        st.subheader("Country Average Levels")
        country_avg = df.groupby('Country')[all_features].mean().reset_index()
        st.dataframe(country_avg)
        
        pollution_summary = country_avg[['Country']].copy()
        for pollutant in pollutants:
            pollution_summary[pollutant+'_Level'] = pd.qcut(country_avg[pollutant], q=3, labels=['Low', 'Moderate', 'High'])
        st.dataframe(pollution_summary)


elif page == "AQI Calculation":
    st.header("AQI Calculation (US EPA Standard)")
    df = df_raw.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df = df.sort_values(by='Date')
    df.drop('Date', axis=1, inplace=True)

    df['AQI_PM25'] = df['PM2.5'].apply(aqi_pm25)
    df['AQI_PM10'] = df['PM10'].apply(aqi_pm10)
    df['AQI_NO2'] = df['NO2'].apply(aqi_no2)
    df['AQI_SO2'] = df['SO2'].apply(aqi_so2)
    df['AQI_CO'] = df['CO'].apply(aqi_co)
    df['AQI_O3'] = df['O3'].apply(aqi_o3)
    df['AQI'] = df[['AQI_PM25', 'AQI_PM10', 'AQI_NO2', 'AQI_SO2', 'AQI_CO', 'AQI_O3']].max(axis=1)
    df['AQI_CATEGORY'] = df['AQI'].apply(aqi_category)
    df.drop(columns=[f'AQI_{p}' for p in ['PM25','PM10','NO2','SO2','CO','O3']], inplace=True)

    st.success("AQI and AQI_CATEGORY computed!")
    st.dataframe(df[['AQI', 'AQI_CATEGORY']].head(10))
    fig = px.histogram(df, x='AQI_CATEGORY', title="Distribution of AQI Categories", color='AQI_CATEGORY')
    st.plotly_chart(fig, use_container_width=True)

elif page == "Regression Models for AQI Prediction":
    st.header("Regression: Predict AQI Value")

    df = df_raw.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df.drop('Date', axis=1, inplace=True)

    # Compute AQI
    for pollutant in ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']:
        df[f'AQI_{pollutant.replace(".","")}'] = df[pollutant].apply(globals()[f'aqi_{pollutant.lower().replace(".","")}'])
    df['AQI'] = df.filter(like='AQI_').max(axis=1)
    df['AQI_CATEGORY'] = df['AQI'].apply(aqi_category)

    X = df[features]
    y = df['AQI']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    def train_models(_X_train, _y_train):
        rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        xgb_model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
        rf.fit(_X_train, _y_train)
        xgb_model.fit(_X_train, _y_train)
        return rf, xgb_model, scaler

    rf_model, xgb_model, trained_scaler = train_models(X_train_scaled, y_train)

    # Save to session state for prediction tool
    st.session_state.rf_regressor = rf_model
    st.session_state.reg_scaler = trained_scaler
    st.session_state.features = features

    tab1, tab2 = st.tabs(["Random Forest", "XGBoost"])

    with tab1:
        pred = rf_model.predict(X_test_scaled)
        col1, col2, col3 = st.columns(3)
        col1.metric("R²", f"{r2_score(y_test, pred):.4f}")
        col2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, pred)):.2f}")
        col3.metric("MAE", f"{mean_absolute_error(y_test, pred):.2f}")

        pred_cat = np.vectorize(aqi_category)(np.round(pred))
        true_cat = np.vectorize(aqi_category)(y_test.values)
        cm = confusion_matrix(true_cat, pred_cat)
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

        imp = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)
        fig2, ax2 = plt.subplots(figsize=(10,6))
        sns.barplot(x=imp.values, y=imp.index, ax=ax2, palette="viridis")
        st.pyplot(fig2)

    with tab2:
        pred = xgb_model.predict(X_test_scaled)
        col1, col2, col3 = st.columns(3)
        col1.metric("R²", f"{r2_score(y_test, pred):.4f}")
        col2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, pred)):.2f}")
        col3.metric("MAE", f"{mean_absolute_error(y_test, pred):.2f}")

        pred_cat = np.vectorize(aqi_category)(np.round(pred))
        true_cat = np.vectorize(aqi_category)(y_test.values)
        cm = confusion_matrix(true_cat, pred_cat)
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax)
        st.pyplot(fig)

        imp = pd.Series(xgb_model.feature_importances_, index=features).sort_values(ascending=False)
        fig2, ax2 = plt.subplots(figsize=(10,6))
        sns.barplot(x=imp.values, y=imp.index, ax=ax2, palette="plasma")
        st.pyplot(fig2)

# ====================== Classification Models for AQI Category ======================
elif page == "Classification Models for AQI Category":
    st.header("Classification Models: Predicting AQI Category")

    df = df_raw.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df.drop('Date', axis=1, inplace=True)

    # Compute AQI using global functions
    pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
    for p in pollutants:
        func_name = f"aqi_{p.lower().replace('.', '')}"
        df[f'AQI_{p.replace(".", "")}'] = df[p].apply(globals()[func_name])
    df['AQI'] = df.filter(like='AQI_').max(axis=1)
    df['AQI_CATEGORY'] = df['AQI'].apply(aqi_category)

    X = df[features]
    y = df['AQI_CATEGORY']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Cache models
    def train_logistic():
        return LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)

    def train_rf_clf():
        return RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)

    def train_xgb_clf():
        return xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            eval_metric='mlogloss'
        )

    def train_knn():
        return KNeighborsClassifier(n_neighbors=6, weights='uniform')

    tab1, tab2, tab3, tab4 = st.tabs(["Logistic Regression", "Random Forest", "XGBoost", "KNN"])

    def evaluate_model(model, name, color):
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        pred_labels = le.inverse_transform(pred)
        true_labels = le.inverse_transform(y_test)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Classification Report")
            st.text(classification_report(true_labels, pred_labels))

        with col2:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(true_labels, pred_labels)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap=color, ax=ax,
                        xticklabels=le.classes_, yticklabels=le.classes_)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

        # Feature importance where available
        if hasattr(model, 'feature_importances_'):
            imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
            st.subheader("Feature Importance")
            fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
            sns.barplot(x=imp.values, y=imp.index, palette="viridis", ax=ax_imp)
            ax_imp.set_title(f"Feature Importance - {name}")
            st.pyplot(fig_imp)
        elif name == "KNN":
            # Permutation importance for KNN
            result = permutation_importance(model, X_test_scaled, y_test, n_repeats=10, random_state=42)
            imp = pd.Series(result.importances_mean, index=features).sort_values(ascending=False)
            st.subheader("Feature Importance (Permutation)")
            fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
            sns.barplot(x=imp.values, y=imp.index, palette="plasma", ax=ax_imp)
            ax_imp.set_title("Feature Importance - KNN")
            st.pyplot(fig_imp)

    with tab1:
        st.subheader("Logistic Regression")
        model = train_logistic()
        evaluate_model(model, "Logistic Regression", "Purples")

    with tab2:
        st.subheader("Random Forest Classifier")
        model = train_rf_clf()
        evaluate_model(model, "Random Forest", "Blues")

    with tab3:
        st.subheader("XGBoost Classifier")
        model = train_xgb_clf()
        evaluate_model(model, "XGBoost", "Oranges")

    with tab4:
        st.subheader("K-Nearest Neighbors")
        model = train_knn()
        evaluate_model(model, "KNN", "Reds")
elif page == "Prediction Tool":
    st.header("Live AQI Prediction")

    if 'rf_regressor' not in st.session_state:
        st.warning("Please visit 'Regression Models' page first to train the model.")
        st.stop()

    # Preprocess df_raw to include Month and Day
    df = df_raw.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day

    st.write("Enter values to predict AQI:")
    inputs = {}
    for f in st.session_state.features:
        inputs[f] = st.number_input(f, value=float(df[f].mean()), step=0.1)

    if st.button("Predict"):
        input_df = pd.DataFrame([inputs])
        input_scaled = st.session_state.reg_scaler.transform(input_df)
        pred_aqi = st.session_state.rf_regressor.predict(input_scaled)[0]
        pred_cat = aqi_category(pred_aqi)
        st.success(f"**Predicted AQI: {pred_aqi:.1f} → {pred_cat}**")
elif page == "Feature Importance Summary":
    st.header("Feature Importance Summary with AQI Analysis")
    
    # Load and preprocess data with AQI calculation
    df = df_raw.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df = df.sort_values(by='Date', ascending=True)
    
    # Calculate AQI
    df['AQI_PM25'] = df['PM2.5'].apply(aqi_pm25)
    df['AQI_PM10'] = df['PM10'].apply(aqi_pm10)
    df['AQI_NO2'] = df['NO2'].apply(aqi_no2)
    df['AQI_SO2'] = df['SO2'].apply(aqi_so2)
    df['AQI_CO'] = df['CO'].apply(aqi_co)
    df['AQI_O3'] = df['O3'].apply(aqi_o3)
    df['AQI'] = df[['AQI_PM25', 'AQI_PM10', 'AQI_NO2', 'AQI_SO2', 'AQI_CO', 'AQI_O3']].max(axis=1)
    df['AQI_CATEGORY'] = df['AQI'].apply(aqi_category)
    
    # Drop intermediate columns
    df.drop(columns=[f'AQI_{p}' for p in ['PM25','PM10','NO2','SO2','CO','O3']], inplace=True)
    df.drop('Date', axis=1, inplace=True)
    
    # Add month name for visualization
    month_map = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
                 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
    df['Month_Name'] = df['Month'].map(month_map)
    
    # Define features (including AQI)
    all_features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 
                    'Temperature', 'Humidity', 'Wind Speed', 'Month', 'Day', 'AQI']
    pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
    meteo_vars = ['Temperature', 'Humidity', 'Wind Speed']
    
    # Color palette
    colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)', 
              'rgb(214, 39, 40)', 'rgb(148, 103, 189)', 'rgb(140, 86, 75)', 
              'rgb(227, 119, 194)', 'rgb(127, 127, 127)', 'rgb(188, 189, 34)',
              'rgb(23, 190, 207)', 'rgb(255, 187, 120)', 'rgb(144, 103, 167)']
    
    # Create tabs for different analyses
    tab_overview, tab_correlation, tab_geographic, tab_seasonal, tab_importance = st.tabs([
        "AQI Overview", "Correlation Analysis", "Geographic Patterns", 
        "Seasonal Trends", "Feature Importance"
    ])
    
    # ========== TAB 1: AQI Overview ==========
    with tab_overview:
        st.subheader("AQI Distribution and Statistics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average AQI", f"{df['AQI'].mean():.1f}")
        with col2:
            st.metric("Median AQI", f"{df['AQI'].median():.1f}")
        with col3:
            st.metric("Max AQI", f"{df['AQI'].max():.1f}")
        
        # AQI Category Distribution
        st.subheader("AQI Category Distribution")
        aqi_cat_counts = df['AQI_CATEGORY'].value_counts()
        col1, col2 = st.columns(2)
        with col1:
            fig_bar = px.bar(x=aqi_cat_counts.index, y=aqi_cat_counts.values,
                           title="AQI Categories Distribution",
                           color=aqi_cat_counts.index,
                           labels={'x': 'AQI Category', 'y': 'Count'})
            st.plotly_chart(fig_bar, use_container_width=True)
        with col2:
            fig_pie = px.pie(names=aqi_cat_counts.index, values=aqi_cat_counts.values,
                           title="AQI Categories Proportion")
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # AQI Histogram
        st.subheader("AQI Value Distribution")
        fig_hist = px.histogram(df, x='AQI', nbins=50, 
                               title="Distribution of AQI Values",
                               color_discrete_sequence=['rgb(31, 119, 180)'])
        fig_hist.update_layout(bargap=0.1)
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Box plot of AQI
        fig_box = px.box(df, y='AQI', title="Box Plot of AQI Values",
                        color_discrete_sequence=['rgb(255, 127, 14)'])
        st.plotly_chart(fig_box, use_container_width=True)
    
    # ========== TAB 2: Correlation Analysis ==========
    with tab_correlation:
        st.subheader("Correlation with AQI")
        
        # Calculate correlations with AQI
        correlation_with_aqi = {}
        for feature in pollutants + meteo_vars:
            correlation = df[feature].corr(df['AQI'])
            correlation_with_aqi[feature] = correlation
        
        # Create correlation dataframe
        corr_df = pd.DataFrame({
            'Feature': list(correlation_with_aqi.keys()),
            'Correlation_with_AQI': list(correlation_with_aqi.values())
        }).sort_values('Correlation_with_AQI', ascending=False)
        
        # Display correlation table
        st.dataframe(corr_df, use_container_width=True)
        
        # Correlation bar chart
        fig_corr_bar = px.bar(corr_df, x='Correlation_with_AQI', y='Feature',
                             title='Feature Correlation with AQI',
                             color='Correlation_with_AQI',
                             color_continuous_scale='RdBu_r',
                             orientation='h')
        fig_corr_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_corr_bar, use_container_width=True)
        
        # Scatter plots of top correlated features with AQI
        st.subheader("Scatter Plots: Features vs AQI")
        top_features = corr_df.head(3)['Feature'].tolist()
        
        for feature in top_features:
            fig_scatter = px.scatter(df, x=feature, y='AQI',
                                   title=f'{feature} vs AQI',
                                   trendline='ols',
                                   color_discrete_sequence=['rgb(44, 160, 44)'])
            fig_scatter.update_traces(marker=dict(size=5, opacity=0.6))
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Full correlation matrix
        st.subheader("Full Correlation Matrix")
        corr_matrix = df[pollutants + meteo_vars + ['AQI']].corr()
        fig_corr_matrix = px.imshow(corr_matrix, 
                                   text_auto=True, 
                                   color_continuous_scale='RdBu_r',
                                   zmin=-1, zmax=1,
                                   title="Correlation Matrix (Including AQI)")
        fig_corr_matrix.update_layout(width=800, height=800)
        st.plotly_chart(fig_corr_matrix, use_container_width=True)
    
    # ========== TAB 3: Geographic Patterns ==========
    with tab_geographic:
        st.subheader("AQI by Country")
        
        # Average AQI by country
        aqi_by_country = df.groupby('Country')['AQI'].agg(['mean', 'std', 'count']).reset_index()
        aqi_by_country = aqi_by_country.sort_values('mean', ascending=False)
        
        col1, col2 = st.columns(2)
        with col1:
            fig_country_bar = px.bar(aqi_by_country, x='Country', y='mean',
                                    error_y='std',
                                    title='Average AQI by Country',
                                    color='mean',
                                    color_continuous_scale='RdYlGn_r')
            fig_country_bar.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_country_bar, use_container_width=True)
        
        with col2:
            fig_country_box = px.box(df, x='Country', y='AQI',
                                    title='AQI Distribution by Country',
                                    color='Country')
            fig_country_box.update_layout(xaxis_tickangle=45, showlegend=False)
            st.plotly_chart(fig_country_box, use_container_width=True)
        
        # AQI by City
        st.subheader("AQI by City")
        aqi_by_city = df.groupby('City')['AQI'].agg(['mean', 'std', 'count']).reset_index()
        aqi_by_city = aqi_by_city.sort_values('mean', ascending=False).head(20)
        
        fig_city = px.bar(aqi_by_city, x='City', y='mean',
                         title='Top 20 Cities by Average AQI',
                         color='mean',
                         color_continuous_scale='RdYlGn_r')
        fig_city.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig_city, use_container_width=True)
        
        # Geographic distribution of AQI categories
        st.subheader("AQI Category Distribution by Country")
        aqi_cat_by_country = df.groupby(['Country', 'AQI_CATEGORY']).size().reset_index(name='count')
        fig_stacked = px.bar(aqi_cat_by_country, x='Country', y='count',
                            color='AQI_CATEGORY',
                            title='AQI Category Distribution by Country',
                            barmode='stack')
        fig_stacked.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig_stacked, use_container_width=True)
    
    # ========== TAB 4: Seasonal Trends ==========
    with tab_seasonal:
        st.subheader("Monthly AQI Trends")
        
        # AQI by month
        monthly_aqi = df.groupby(['Month_Name', 'Month'])['AQI'].agg(['mean', 'std', 'count']).reset_index()
        monthly_aqi = monthly_aqi.sort_values('Month')
        
        fig_month = px.line(monthly_aqi, x='Month_Name', y='mean',
                           error_y='std',
                           title='Monthly AQI Trends',
                           markers=True)
        fig_month.update_traces(line=dict(width=3))
        st.plotly_chart(fig_month, use_container_width=True)
        
        # Heatmap: Month vs AQI Category
        st.subheader("Monthly AQI Category Distribution")
        month_cat_heatmap = pd.crosstab(df['Month_Name'], df['AQI_CATEGORY'])
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        month_cat_heatmap = month_cat_heatmap.reindex(month_order)
        
        fig_heatmap = px.imshow(month_cat_heatmap.T,
                               labels=dict(x="Month", y="AQI Category", color="Count"),
                               color_continuous_scale='YlOrRd',
                               title="Monthly Distribution of AQI Categories")
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Seasonal variation of pollutants and their impact on AQI
        st.subheader("Pollutant Contribution to AQI by Month")
        monthly_pollutants = df.groupby('Month_Name')[pollutants].mean().reindex(month_order)
        monthly_aqi_mean = df.groupby('Month_Name')['AQI'].mean().reindex(month_order)
        
        fig_combo = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add pollutant lines
        for pollutant in pollutants:
            fig_combo.add_trace(
                go.Scatter(x=monthly_pollutants.index, y=monthly_pollutants[pollutant],
                          name=pollutant, mode='lines+markers'),
                secondary_y=False
            )
        
        # Add AQI line
        fig_combo.add_trace(
            go.Scatter(x=monthly_aqi_mean.index, y=monthly_aqi_mean.values,
                      name="AQI", mode='lines+markers', line=dict(width=4)),
            secondary_y=True
        )
        
        fig_combo.update_layout(
            title="Pollutant Concentrations and AQI by Month",
            xaxis_title="Month",
            height=500
        )
        fig_combo.update_yaxes(title_text="Pollutant Concentration", secondary_y=False)
        fig_combo.update_yaxes(title_text="AQI", secondary_y=True)
        st.plotly_chart(fig_combo, use_container_width=True)
    
    # ========== TAB 5: Feature Importance ==========
    with tab_importance:
        st.subheader("Statistical Feature Importance Analysis")
        
        # Train a model to get feature importance
        X = df[features]  # Original features without AQI
        y = df['AQI']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest for feature importance
    
        def get_feature_importance():
            rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
            rf.fit(X_train_scaled, y_train)
            return rf
        
        rf_model = get_feature_importance()
        
        # Get feature importances
        importances = pd.Series(rf_model.feature_importances_, index=features)
        importances = importances.sort_values(ascending=False)
        
        # Display feature importance
        st.subheader("Random Forest Feature Importance for AQI Prediction")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            fig_importance = px.bar(x=importances.values, y=importances.index,
                                   orientation='h',
                                   title='Feature Importance for AQI Prediction',
                                   color=importances.values,
                                   color_continuous_scale='Viridis',
                                   labels={'x': 'Importance Score', 'y': 'Feature'})
            fig_importance.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_importance, use_container_width=True)
        
        with col2:
            st.write("**Importance Scores:**")
            importance_df = pd.DataFrame({
                'Feature': importances.index,
                'Importance': importances.values
            })
            st.dataframe(importance_df, use_container_width=True)
        
        # SHAP-like analysis using permutation importance
        st.subheader("Permutation Importance Analysis")
        
        from sklearn.inspection import permutation_importance
        perm_importance = permutation_importance(
            rf_model, X_test_scaled, y_test, n_repeats=10, random_state=42
        )
        
        perm_importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': perm_importance.importances_mean,
            'Std': perm_importance.importances_std
        }).sort_values('Importance', ascending=False)
        
        fig_perm = px.bar(perm_importance_df, x='Importance', y='Feature',
                         error_x='Std',
                         orientation='h',
                         title='Permutation Importance',
                         color='Importance',
                         color_continuous_scale='Plasma')
        fig_perm.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_perm, use_container_width=True)
        
        # Partial dependence analysis for top features
        st.subheader("Partial Dependence Analysis")
        top_3_features = importances.head(3).index.tolist()
        
        for feature in top_3_features:
            # Create bins for the feature
            df['feature_bin'] = pd.qcut(df[feature], q=10, duplicates='drop')
            bin_avg = df.groupby('feature_bin').agg({'AQI': 'mean', feature: 'mean'}).reset_index()
            
            fig_pd = px.scatter(bin_avg, x=feature, y='AQI',
                               title=f'Partial Dependence: {feature} → AQI',
                               trendline='lowess',
                               labels={feature: f'{feature} (binned)', 'AQI': 'Average AQI'})
            fig_pd.update_traces(marker=dict(size=10))
            st.plotly_chart(fig_pd, use_container_width=True)
        
        # Feature interactions
        st.subheader("Feature Interactions Impact on AQI")
        
        if len(top_3_features) >= 2:
            # Create interaction plot for top 2 features
            feature1, feature2 = top_3_features[0], top_3_features[1]
            
            # Create bins for both features
            df['feat1_bin'] = pd.qcut(df[feature1], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            df['feat2_bin'] = pd.qcut(df[feature2], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            
            interaction_df = df.groupby(['feat1_bin', 'feat2_bin'])['AQI'].mean().reset_index()
            
            fig_interaction = px.density_heatmap(
                interaction_df, x='feat1_bin', y='feat2_bin', z='AQI',
                title=f'Interaction: {feature1} × {feature2} → AQI',
                color_continuous_scale='RdYlGn_r',
                labels={'feat1_bin': feature1, 'feat2_bin': feature2, 'AQI': 'Average AQI'}
            )
            st.plotly_chart(fig_interaction, use_container_width=True)
        
        # Summary insights
        st.subheader("Key Insights")
        
        insights = f"""
        ### Top 3 Most Important Features for AQI:
        1. **{importances.index[0]}**: {importances.values[0]:.2%} of predictive power
        2. **{importances.index[1]}**: {importances.values[1]:.2%} of predictive power  
        3. **{importances.index[2]}**: {importances.values[2]:.2%} of predictive power
        
        ### Key Findings:
        - **{importances.index[0]}** shows the strongest correlation with AQI
        - Meteorological factors contribute approximately {(importances[['Temperature', 'Humidity', 'Wind Speed']].sum()*100):.1f}% to AQI prediction
        - Temporal features (Month, Day) contribute {(importances[['Month', 'Day']].sum()*100):.1f}% to AQI prediction
        - The top 3 features account for {(importances.head(3).sum()*100):.1f}% of total predictive power
        """
        
        st.markdown(insights)
