import os
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# تحديد المسارات الأساسية لملفات البيانات لضمان العمل من أي دليل تشغيل
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_COVID_DATA_PATH = os.path.join(BASE_DIR, 'cleaned_covid_data_last.csv')
DEFAULT_VACCINE_DATA_PATH = os.path.join(BASE_DIR, 'country_vaccinations.csv')


# 1. تحميل البيانات
@st.cache_data
def load_data(filepath=None):
    """
    تحميل بيانات كوفيد-19 المعالجة مع ضبط أنواع البيانات والتاريخ.
    """
    if filepath is None:
        filepath = DEFAULT_COVID_DATA_PATH if os.path.exists(DEFAULT_COVID_DATA_PATH) else 'cleaned_covid_data_last.csv'
        
    df = pd.read_csv(filepath,
                     parse_dates=['date'],
                     dtype={'new_cases': 'float32',
                            'total_cases': 'float32',
                            'population': 'float32',
                            'stringency_index': 'float32'})
    return df


@st.cache_data
def load_vaccine_data(filepath=None):
    """
    تحميل بيانات التلقيح العالمية مع تصفية البيانات بدءاً من عام 2021.
    """
    if filepath is None:
        filepath = DEFAULT_VACCINE_DATA_PATH if os.path.exists(DEFAULT_VACCINE_DATA_PATH) else 'country_vaccinations.csv'

    if not os.path.exists(filepath):
        return None

    try:
        vaccine_df = pd.read_csv(filepath, parse_dates=['date'])
        vaccine_df = vaccine_df[vaccine_df['date'].dt.year >= 2021]
        vaccine_df['country'] = vaccine_df['country'].str.strip()
        return vaccine_df
    except Exception:
        return None


# 2. معالجة البيانات
def fit_outlier_bounds(train_df, lower_quantile=0.05, upper_quantile=0.95):
    """
    حساب حدود القيم المتطرفة من بيانات التدريب فقط لمنع تسريب البيانات.
    """
    q_low = float(train_df['new_cases'].quantile(lower_quantile))
    q_high = float(train_df['new_cases'].quantile(upper_quantile))
    return q_low, q_high


def apply_outlier_bounds(df, q_low, q_high):
    """
    تطبيق حدود القيم المتطرفة المتعلمة من بيانات التدريب على أي مجموعة بيانات.
    """
    df = df.copy()
    return df[(df['new_cases'] >= q_low) & (df['new_cases'] <= q_high)]


def preprocess_data(df, filter_outliers=True, q_low=None, q_high=None):
    """
    معالجة وتجهيز البيانات:
    - التعامل مع القيم المفقودة بالملء الأمامي والتعويض بالأصفار لكل دولة
    - ضمان عدم وجود قيم سالبة للحالات
    - إزالة القيم المتطرفة عند الطلب (مع دعم تمرير حدود متعلمة مسبقاً من التدريب فقط)
    """
    df = df.copy()
    if 'location' in df.columns and 'date' in df.columns:
        df = df.sort_values(['location', 'date'])
        df['new_cases'] = df.groupby('location')['new_cases'].transform(lambda s: s.ffill().fillna(0))
    else:
        df['new_cases'] = df['new_cases'].ffill().fillna(0)
    
    df['new_cases'] = df['new_cases'].clip(lower=0)
    
    if filter_outliers:
        if q_low is None:
            q_low = float(df['new_cases'].quantile(0.05))
        if q_high is None:
            q_high = float(df['new_cases'].quantile(0.95))
        df = df[(df['new_cases'] >= q_low) & (df['new_cases'] <= q_high)]
        
    return df


def check_date_continuity(df):
    """
    التحقق من استمرارية التواريخ اليومية بدون فجوات زمنية مفقودة لكل دولة.
    """
    if 'date' not in df.columns:
        return True, 0
    df_sorted = df.copy()
    df_sorted['date'] = pd.to_datetime(df_sorted['date'])
    if 'location' in df_sorted.columns:
        df_sorted = df_sorted.sort_values(['location', 'date'])
        diffs = df_sorted.groupby('location')['date'].diff().dt.days
    else:
        df_sorted = df_sorted.sort_values('date')
        diffs = df_sorted['date'].diff().dt.days
    gaps = int((diffs > 1).sum())
    return gaps == 0, gaps


def reindex_daily_calendar(df):
    """
    إعادة فهرسة البيانات لضمان وجود تسلسل يومي مستمر بدون انقطاع لكل دولة،
    مع ملء الأيام المفقودة بالحالات الصفرية والملء الأمامي للمتغيرات التراكمية والديموغرافية.
    """
    if 'date' not in df.columns or 'location' not in df.columns:
        return df
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    dfs = []
    for loc, group in df.groupby('location'):
        group = group.sort_values('date').drop_duplicates(subset=['date'])
        full_idx = pd.date_range(group['date'].min(), group['date'].max(), freq='D')
        group = group.set_index('date').reindex(full_idx)
        group['location'] = loc
        if 'new_cases' in group.columns:
            group['new_cases'] = group['new_cases'].fillna(0.0)
        for col in ['total_cases', 'population', 'stringency_index', 'total_deaths']:
            if col in group.columns:
                group[col] = group[col].ffill().fillna(0.0)
        group = group.reset_index().rename(columns={'index': 'date'})
        dfs.append(group)
        
    return pd.concat(dfs, ignore_index=True).sort_values(['location', 'date'])


# 3. إنشاء الميزات وهندستها
def create_features(df):
    """
    إنشاء ميزات الفارق الزمني (Lag features) والمتوسطات المتحركة (Rolling Averages)
    الخالية تماماً من تسريب البيانات المستهدفة أو الزمنية (Target & Temporal Leakage-Free).
    جميع الميزات المحسوبة لليوم t تعتمد حصرياً على البيانات المتاحة حتى اليوم t-1.
    """
    df = df.copy()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    if 'location' in df.columns and 'date' in df.columns:
        df = df.sort_values(['location', 'date'])
    elif 'date' in df.columns:
        df = df.sort_values('date')
    
    # تأخير المتغيرات التراكمية والحكومية المعاصرة لضمان توافرها عند لحظة التنبؤ t-1
    if 'total_cases' in df.columns:
        if 'location' in df.columns:
            df['total_cases'] = df.groupby('location')['total_cases'].shift(1)
        else:
            df['total_cases'] = df['total_cases'].shift(1)
            
    if 'stringency_index' in df.columns:
        if 'location' in df.columns:
            df['stringency_index'] = df.groupby('location')['stringency_index'].shift(1)
        else:
            df['stringency_index'] = df['stringency_index'].shift(1)
            
    for lag in [1, 7]:
        if 'location' in df.columns:
            df[f'lag_{lag}'] = df.groupby('location')['new_cases'].shift(lag)
            df[f'rolling_avg_{lag}'] = df.groupby('location')['new_cases'].transform(
                lambda s: s.shift(1).rolling(lag, min_periods=1).mean()
            )
        else:
            df[f'lag_{lag}'] = df['new_cases'].shift(lag)
            df[f'rolling_avg_{lag}'] = df['new_cases'].shift(1).rolling(lag, min_periods=1).mean()
        
    df = df.dropna()
    df['target'] = np.log1p(df['new_cases'])
    return df


# 4. نماذج خط الأساس (Baselines)
class NaivePersistenceModel:
    """
    نموذج الاستمرار الساذج (Naive Persistence Baseline):
    يتنبأ بحالات اليوم التالي بناءً على آخر قيمة معروفة لحالات اليوم السابق (y_t = y_{t-1}).
    """
    def __init__(self):
        self.last_value_ = None

    def fit(self, X=None, y=None):
        if y is not None and len(y) > 0:
            self.last_value_ = float(np.asarray(y)[-1])
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame) and 'lag_1' in X.columns:
            return np.asarray(X['lag_1'], dtype=float)
        elif self.last_value_ is not None:
            return np.full(len(X), self.last_value_, dtype=float)
        raise ValueError("Cannot predict with NaivePersistenceModel without 'lag_1' or fitted history.")


class MovingAverageModel:
    """
    نموذج المتوسط المتحرك (Moving Average Baseline):
    يتنبأ بحالات اليوم التالي بناءً على متوسط نافذة ماضية محسوبة حصرياً من الملاحظات السابقة.
    """
    def __init__(self, window=7):
        self.window = window
        self.mean_value_ = None

    def fit(self, X=None, y=None):
        if y is not None and len(y) > 0:
            y_arr = np.asarray(y, dtype=float)
            w = min(len(y_arr), self.window)
            self.mean_value_ = float(np.mean(y_arr[-w:]))
        return self

    def predict(self, X):
        col_name = f'rolling_avg_{self.window}'
        if isinstance(X, pd.DataFrame) and col_name in X.columns:
            return np.asarray(X[col_name], dtype=float)
        elif self.mean_value_ is not None:
            return np.full(len(X), self.mean_value_, dtype=float)
        raise ValueError(f"Cannot predict with MovingAverageModel without '{col_name}' or fitted history.")


def predict_naive_persistence(X):
    """
    تنبؤ الاستمرار الساذج لليوم التالي باستخدام lag_1.
    """
    if isinstance(X, pd.DataFrame) and 'lag_1' in X.columns:
        return np.asarray(X['lag_1'], dtype=float)
    raise ValueError("X must be a DataFrame containing 'lag_1'")


def predict_moving_average(X, window=7):
    """
    تنبؤ المتوسط المتحرك لليوم التالي من الماضي فقط باستخدام rolling_avg_{window}.
    """
    col_name = f'rolling_avg_{window}'
    if isinstance(X, pd.DataFrame) and col_name in X.columns:
        return np.asarray(X[col_name], dtype=float)
    raise ValueError(f"X must be a DataFrame containing '{col_name}'")


# 4. بناء نموذج التنبؤ
def build_model(n_estimators=500, learning_rate=0.05, max_depth=7, random_state=42):
    """
    تهيئة نموذج Gradient Boosting Regressor بالمعاملات الفائقة المحددة.
    """
    return GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=random_state,
        loss='huber',
        alpha=0.95,
        subsample=0.8,
        validation_fraction=0.1,
        n_iter_no_change=5
    )


# 5. حساب SMAPE المعدل
def adjusted_smape(y_true, y_pred, threshold=1.0):
    """
    حساب مقياس الخطأ النسبي المتماثل المعدل (Adjusted SMAPE).
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    denominator = (np.abs(y_true_arr) + np.abs(y_pred_arr))
    mask = denominator > threshold
    if np.sum(mask) == 0:
        return np.nan
    smape_value = 100 / np.sum(mask) * np.sum(2 * np.abs(y_pred_arr[mask] - y_true_arr[mask]) / denominator[mask])
    return float(smape_value)


# 6. حساب مقاييس التقييم الشاملة
def evaluate_predictions(y_true, y_pred):
    """
    حساب مقاييس الأداء: MAE, RMSE, R2, Adjusted SMAPE, Accuracy.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    
    mae = mean_absolute_error(y_true_arr, y_pred_arr)
    rmse = np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))
    r2 = r2_score(y_true_arr, y_pred_arr)
    smape = adjusted_smape(y_true_arr, y_pred_arr)
    accuracy_estimate = 100 - smape if not np.isnan(smape) else np.nan
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'SMAPE': smape,
        'Accuracy': accuracy_estimate
    }


def main():
    # صفحة إعدادات Streamlit
    st.set_page_config(page_title="COVID-19 Prediction Model", layout="wide", page_icon="🦠")

    # عنوان التطبيق
    st.title('🦠 COVID-19 Cases Prediction Model')
    st.markdown("""
    This app uses Gradient Boosting to predict COVID-19 cases based on historical data.
    """)

    # شريط جانبي للتحكم
    st.sidebar.header('Model Controls')

    with st.spinner('Loading data...'):
        df_raw = load_data()
        vaccine_df = load_vaccine_data()

    df = preprocess_data(df_raw, filter_outliers=False)

    # اختيار الوظيفة من الشريط الجانبي
    app_mode = st.sidebar.radio("Select Function", ["Model Prediction", "Exploratory Data Analysis"])

    if app_mode == "Model Prediction":
        df = create_features(df)

        # اختيار الميزات
        features = ['lag_1', 'lag_7', 'rolling_avg_1', 'rolling_avg_7',
                    'total_cases', 'population', 'stringency_index']

        # اختيار الدولة من الشريط الجانبي
        countries = df['location'].unique()
        selected_country = st.sidebar.selectbox('Select Country', countries)
        country_df = df[df['location'] == selected_country].copy()

        # تقسيم البيانات
        train_size = st.sidebar.slider('Train/Test Split Ratio', 0.6, 0.9, 0.8, 0.05)
        split_idx = int(len(country_df) * train_size)

        X = country_df[features]
        y = country_df['target']

        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # إعداد النموذج في الشريط الجانبي
        st.sidebar.subheader('Model Parameters')
        n_estimators = st.sidebar.slider('Number of Estimators', 100, 1000, 500, 50)
        learning_rate = st.sidebar.slider('Learning Rate', 0.01, 0.2, 0.05, 0.01)
        max_depth = st.sidebar.slider('Max Depth', 3, 10, 7)

        # زر لتدريب النموذج
        if st.sidebar.button('Train Model'):
            with st.spinner('Training the model...'):
                
                # 6. Actual vs Baseline comparison (Top 5 Countries Baseline)
                st.subheader('Actual vs 7-Day Moving Average Baseline (Top 5 Countries)')

                # إنشاء نسخة مبسطة من البيانات لجميع الدول كمرجع مقارن
                full_results = df[['date', 'location', 'new_cases']].copy()
                full_results = full_results.rename(columns={'location': 'Country', 'date': 'Date', 'new_cases': 'Actual'})
                full_results['7-Day MA Baseline'] = full_results.groupby('Country')['Actual'].transform(
                    lambda s: s.shift(1).rolling(7, min_periods=1).mean()
                )

                top_countries = full_results['Country'].unique()[:5]

                fig = go.Figure()

                for country in top_countries:
                    country_data = full_results[full_results['Country'] == country]
            
                    # Actual cases
                    fig.add_trace(go.Scatter(
                        x=country_data['Date'],
                        y=country_data['Actual'],
                        mode='lines',
                        name=f'{country} - Actual',
                        line=dict(dash='solid')
                    ))
            
                    # 7-Day Moving Average Baseline
                    fig.add_trace(go.Scatter(
                        x=country_data['Date'],
                        y=country_data['7-Day MA Baseline'],
                        mode='lines',
                        name=f'{country} - 7-Day MA Baseline',
                        line=dict(dash='dash')
                    ))

                fig.update_layout(
                    title='Actual vs 7-Day Moving Average Baseline COVID-19 Cases Over Time by Country',
                    xaxis_title='Date',
                    yaxis_title='Number of Cases',
                    legend_title='Legend',
                    height=600,
                    width=1000,
                    template='plotly_white',
                    margin=dict(l=40, r=40, t=60, b=40)
                )

                st.plotly_chart(fig)

                # 7. بناء وتدريب النموذج
                model = build_model(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    random_state=42
                )
                
                model.fit(X_train, y_train)
                
                # 8. التنبؤ والتقييم
                y_pred = np.expm1(model.predict(X_test))
                y_test_actual = np.expm1(y_test)
                
                # حساب نماذج خط الأساس
                naive_model = NaivePersistenceModel().fit(X_train, np.expm1(y_train))
                y_pred_naive = naive_model.predict(X_test)
                metrics_naive = evaluate_predictions(y_test_actual, y_pred_naive)

                ma_model = MovingAverageModel(window=7).fit(X_train, np.expm1(y_train))
                y_pred_ma = ma_model.predict(X_test)
                metrics_ma = evaluate_predictions(y_test_actual, y_pred_ma)

                # حساب المقاييس للنموذج الأساسي
                metrics = evaluate_predictions(y_test_actual, y_pred)
                
                # عرض النتائج
                st.success('Model training completed!')
                
                # جدول مقارنة النماذج مع خطوط الأساس
                st.subheader('Model & Baseline Evaluation Comparison')
                comparison_df = pd.DataFrame([
                    {
                        'Model': 'Gradient Boosting Regressor',
                        'MAE': metrics['MAE'],
                        'RMSE': metrics['RMSE'],
                        'R² Score': metrics['R2'],
                        'Adjusted SMAPE': f"{metrics['SMAPE']:.2f}%" if not np.isnan(metrics['SMAPE']) else "N/A"
                    },
                    {
                        'Model': '7-Day Moving Average Baseline',
                        'MAE': metrics_ma['MAE'],
                        'RMSE': metrics_ma['RMSE'],
                        'R² Score': metrics_ma['R2'],
                        'Adjusted SMAPE': f"{metrics_ma['SMAPE']:.2f}%" if not np.isnan(metrics_ma['SMAPE']) else "N/A"
                    },
                    {
                        'Model': 'Naive Persistence Baseline (Lag 1)',
                        'MAE': metrics_naive['MAE'],
                        'RMSE': metrics_naive['RMSE'],
                        'R² Score': metrics_naive['R2'],
                        'Adjusted SMAPE': f"{metrics_naive['SMAPE']:.2f}%" if not np.isnan(metrics_naive['SMAPE']) else "N/A"
                    }
                ])
                st.dataframe(comparison_df.style.format({
                    'MAE': '{:.2f}',
                    'RMSE': '{:.2f}',
                    'R² Score': '{:.4f}'
                }))

                # عرض المقاييس في أعمدة
                col1, col2, col3 = st.columns(3)
                col1.metric("MAE", f"{metrics['MAE']:.2f}")
                col2.metric("RMSE", f"{metrics['RMSE']:.2f}")
                col3.metric("R² Score", f"{metrics['R2']:.4f}")
                
                # إنشاء DataFrame للنتائج
                results = pd.DataFrame({
                    'Date': country_df.loc[X_test.index, 'date'],
                    'Actual': y_test_actual,
                    'Predicted': y_pred,
                    '7-Day MA Baseline': y_pred_ma,
                    'Naive Baseline': y_pred_naive
                })
                
                # عرض أول 10 تنبؤات
                st.subheader('Sample Predictions')
                st.dataframe(results.head(10))
                
                # 1. الرسم البياني الفعلي vs المتوقع
                fig_scatter = px.scatter(x=y_test_actual, y=y_pred, 
                                        labels={'x': 'Actual Cases', 'y': 'Predicted Cases'},
                                        title='Actual vs Predicted Cases')
                fig_scatter.add_trace(go.Scatter(x=[y_test_actual.min(), y_test_actual.max()],
                                                y=[y_test_actual.min(), y_test_actual.max()],
                                                mode='lines', name='Perfect Prediction'))
                st.plotly_chart(fig_scatter)
                
                # 2. توزيع الأخطاء
                errors = y_pred - y_test_actual
                fig_hist = px.histogram(errors, nbins=50, 
                                        title='Prediction Error Distribution',
                                        labels={'value': 'Prediction Error'})
                fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig_hist)
                
                # 3. أهمية الميزات
                if hasattr(model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'Feature': features,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=True)
                    
                    fig_imp = px.bar(importance_df, x='Importance', y='Feature', 
                                    orientation='h', title='Feature Importance')
                    st.plotly_chart(fig_imp)
                
                # 4. الرسم البياني الزمني
                results_df = pd.DataFrame({
                    'Date': country_df.iloc[split_idx:]['date'],
                    'Actual': y_test_actual,
                    'Predicted': y_pred,
                    '7-Day MA Baseline': y_pred_ma,
                    'Naive Baseline': y_pred_naive
                })
                
                fig_line = px.line(results_df, x='Date', y=['Actual', 'Predicted', '7-Day MA Baseline', 'Naive Baseline'],
                                  title=f'COVID-19 Cases in {selected_country} — Forecast vs Baselines')
                st.plotly_chart(fig_line)
                
                # عرض SMAPE ودقة التنبؤ
                smape_score = metrics['SMAPE']
                accuracy_estimate = metrics['Accuracy']
                
                col1, col2 = st.columns(2)
                col1.metric("Adjusted SMAPE", f"{smape_score:.2f}%" if not np.isnan(smape_score) else "N/A")
                col2.metric("Accuracy Estimate", f"{accuracy_estimate:.2f}%" if not np.isnan(accuracy_estimate) else "N/A")
                
                # الدقة للحالات الكبيرة فقط
                large_cases_mask = y_test_actual > 10
                if np.sum(large_cases_mask) > 0:
                    y_test_large = np.asarray(y_test_actual)[large_cases_mask.values]
                    y_pred_large = np.asarray(y_pred)[large_cases_mask.values]
                    accuracy_large = 100 - mean_absolute_percentage_error(y_test_large, y_pred_large) * 100
                    st.metric("Accuracy for cases > 10", f"{accuracy_large:.2f}%")
                
                # 5. تحليل الخطأ النسبي
                st.subheader('Relative Error Analysis')
                relative_errors = (y_pred - np.asarray(y_test_actual)) / (np.asarray(y_test_actual) + 1e-6)
                fig_err, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(y_test_actual, relative_errors, alpha=0.6, color='green')
                ax.axhline(y=0, color='red', linestyle='--')
                ax.set_title('Relative Error vs Actual', fontsize=14)
                ax.set_xlabel('Actual Value', fontsize=12)
                ax.set_ylabel('Relative Error', fontsize=12)
                ax.grid(True)
                st.pyplot(fig_err)
                plt.close(fig_err)

    else:  # EDA Mode
        st.header("Exploratory Data Analysis (EDA)")
        
        # Basic statistics
        st.subheader("Basic Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Data Shape:")
            st.write(f"Number of rows: {df.shape[0]}")
            st.write(f"Number of columns: {df.shape[1]}")
            
        with col2:
            st.write("Data Types:")
            st.write(df.dtypes.astype(str))
        
        st.subheader("Descriptive Statistics")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        desc_stats = df[numeric_cols].describe().transpose()
        desc_stats['count'] = desc_stats['count'].astype(int)
        
        st.dataframe(desc_stats.style.format({
            'mean': '{:.2f}',
            'std': '{:.2f}',
            'min': '{:.2f}',
            '25%': '{:.2f}',
            '50%': '{:.2f}',
            '75%': '{:.2f}',
            'max': '{:.2f}'
        }))
        
        st.subheader("Missing Values")
        st.dataframe(df.isnull().sum().sort_values(ascending=False).rename("Missing Values"))
        
        # Time series analysis
        st.subheader("Time Series Analysis")
        weekly = df.groupby([pd.Grouper(key='date', freq='W'), 'location']).agg({
            'new_cases': 'sum',
            'new_deaths': 'sum',
            'stringency_index': 'mean'
        }).reset_index()
        
        fig = px.line(weekly, x='date', y='new_cases', color='location',
                      title='Weekly Trend of New Cases by Country',
                      labels={'new_cases': 'New Cases', 'date': 'Date'},
                      template='plotly_dark')
        fig.update_layout(hovermode='x unified')
        st.plotly_chart(fig)
        
        # Correlation analysis
        st.subheader("Correlation Analysis")
        df['cases_per_1000'] = (df['new_cases'] / df['population']) * 1000
        df['deaths_per_1000'] = (df['new_deaths'] / df['population']) * 1000
        
        corr_matrix = df[['new_cases', 'new_deaths', 'stringency_index', 
                          'case_fatality_rate', 'population']].corr()
        
        fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                             title='Correlation Matrix of Key Indicators')
        st.plotly_chart(fig_corr)
        
        # Mortality rate analysis
        mortality = df.groupby('location')['case_fatality_rate'].mean().sort_values(ascending=False)
        fig_mort = px.bar(mortality.head(20), orientation='h',
                          title='Top 20 Countries by Average Fatality Rate',
                          labels={'value': 'Fatality Rate', 'index': 'Country'})
        st.plotly_chart(fig_mort)
        
        # Geographic heatmap
        st.subheader("Geographic Heatmap")
        country_data = df.groupby('location').agg({
            'total_cases': 'max',
            'total_deaths': 'max',
            'population': 'max'
        }).reset_index()
        
        country_data['cases_per_100k'] = (country_data['total_cases'] / country_data['population']) * 100000
        country_data['deaths_per_100k'] = (country_data['total_deaths'] / country_data['population']) * 100000
        
        fig_geo = px.choropleth(country_data, 
                                locations="location",
                                locationmode='country names',
                                color="cases_per_100k",
                                hover_name="location",
                                hover_data=["total_cases", "total_deaths"],
                                color_continuous_scale=px.colors.sequential.Plasma,
                                title="Global COVID-19 Spread (Cases per 100k Population)")
        st.plotly_chart(fig_geo)
        
        # Cases vs Deaths relationship
        st.subheader("Cases vs Deaths Relationship")
        fig_scatter_cd = px.scatter(df, x='total_cases', y='total_deaths', color='location',
                                    size='population', log_x=True, log_y=True,
                                    title='Relationship Between Total Cases and Deaths (Log Scale)',
                                    labels={'total_cases': 'Total Cases', 'total_deaths': 'Total Deaths'})
        st.plotly_chart(fig_scatter_cd)
        
        # Calculate fatality rate over time (تمت معالجة القسمة على الصفر واستخدام التردد المحدث لتفادي التحذيرات)
        df['fatality_rate'] = np.where(df['total_cases'] > 0, df['total_deaths'] / df['total_cases'], np.nan)
        # Use MonthEnd ('ME') if available in pandas, otherwise 'M'
        freq_month = 'ME' if hasattr(pd.offsets, 'MonthEnd') else 'M'
        valid_fatality = df.dropna(subset=['fatality_rate'])
        fatality_over_time = valid_fatality.groupby(pd.Grouper(key='date', freq=freq_month))['fatality_rate'].mean().reset_index()
        
        fig_fatality = px.line(fatality_over_time, x='date', y='fatality_rate',
                               title='Evolution of Fatality Rate Over Time (Monthly Average)',
                               labels={'fatality_rate': 'Fatality Rate', 'date': 'Date'})
        st.plotly_chart(fig_fatality)
        
        # Vaccination Analysis (if data available)
        if vaccine_df is not None:
            st.subheader("Vaccination Analysis")
            
            vaccine_time_data = vaccine_df.groupby('date')['total_vaccinations'].sum().reset_index()

            fig_vax_time = px.line(
                vaccine_time_data,
                x='date',
                y='total_vaccinations',
                title='Global Vaccination Rate Over Time',
                labels={'date': 'Date', 'total_vaccinations': 'Total Vaccinations'},
                template='plotly'
            )
            
            fig_vax_time.update_traces(marker=dict(size=4), line=dict(width=2, color='blue'))
            fig_vax_time.update_layout(
                title_font_size=20,
                xaxis_title_font_size=14,
                yaxis_title_font_size=14,
                xaxis=dict(showgrid=True),
                yaxis=dict(showgrid=True)
            )
            st.plotly_chart(fig_vax_time)
            
            # Top 10 countries by vaccinations
            vaccination_summary = vaccine_df.groupby('country')['total_vaccinations'].max().reset_index()
            top_10_vaccinated = vaccination_summary.sort_values('total_vaccinations', ascending=False).head(10)

            fig_vax_bar = px.bar(top_10_vaccinated, 
                                 x='country', 
                                 y='total_vaccinations', 
                                 orientation='v',
                                 title='Top 10 Countries by Total Vaccinations',
                                 labels={'total_vaccinations': 'Total Vaccinations', 'country': 'Country'},
                                 color='total_vaccinations',
                                 color_continuous_scale='Viridis')
            st.plotly_chart(fig_vax_bar)

    # معلومات إضافية
    st.sidebar.markdown("""
    ### Model Information
    - **Algorithm**: Gradient Boosting Regressor
    - **Features Used**: 
      - Lag features (1 day, 7 days)
      - Rolling averages (1 day, 7 days)
      - Total cases, population, stringency index
    """)


if __name__ == "__main__":
    main()