
import plotly.graph_objects as go
import streamlit as st
import seaborn as sns
import plotly.express as px
import pandas as pd
import numpy as np
from funpymodeling.exploratory import freq_tbl 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

try:
    from streamlit_option_menu import option_menu
except ImportError:
    import os
    os.system('pip install streamlit-option-menu')
    from streamlit_option_menu import option_menu

@st.cache_resource
def load_data():
    df = pd.read_csv("MEXICO_LIMPIO.csv")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    df.fillna(method='ffill', inplace=True)
    numeric_df = df.select_dtypes(['float', 'int'])
    numeric_cols = numeric_df.columns 
    return df, numeric_df, numeric_cols

def generate_freq_table(df, column):
    if column in df.columns:
        table = freq_tbl(df[column])
        if isinstance(table, pd.DataFrame):
            if len(table) > 10:
                table = table.head(10)
            return table
        else:
            return None
    else:
        return None

df, numeric_df, numeric_cols = load_data()

if df is None:
    st.stop()

with st.sidebar:
    st.image("1.png", use_column_width=True)
    selected = option_menu(
        menu_title="DASHBOARD",
        options=["Análisis Univariado", "Distribución de Precios", "Correlación y Boxplot", "Gráfico de Dispersión", "Modelos de Regresión"],
        icons=["bar-chart", "pie-chart", "heat-map", "scatter-plot", "line-chart"],
        menu_icon="cast",
        default_index=0,
    )

if selected == "Análisis Univariado":
    st.title("Evaluación de Datos - Análisis Univariado")
    st.header("Análisis Univariado y Gráfico de Barras")
    freq_var = st.sidebar.selectbox("Selecciona una variable para medir su frecuencia", options=df.select_dtypes(include=['object']).columns)
    freq_table = generate_freq_table(df, freq_var)
    st.subheader(f"Tabla de Frecuencia por {freq_var}")
    if freq_table is not None:
        st.write(freq_table)
    else:
        st.error(f"No se pudo generar la tabla de frecuencias para {freq_var}")
    st.subheader(f"Frecuencia por {freq_var}")
    if freq_table is not None:
        barplot_figure = px.bar(freq_table, x=freq_var, y='frequency',
                                title=f'Frecuencia de {freq_var}',
                                color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(barplot_figure)
    st.subheader(f"Gráfico de Pastel por {freq_var}")
    pie_chart = px.pie(freq_table, names=freq_var, values='frequency', title=f'Distribución por {freq_var}', color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(pie_chart)

elif selected == "Distribución de Precios":
    st.title("Evaluación de Datos - Distribución de Precios")
    st.header("Distribución de Precios y Tipos de Habitaciones")
    st.subheader("Histograma del Precio")
    hist_figure = px.histogram(df, x='price', nbins=50, title='Distribución de Precios',
                               color_discrete_sequence=['#636EFA'])
    st.plotly_chart(hist_figure)
    pie_var = st.sidebar.selectbox("Selecciona una variable para el Pie Chart", options=df.select_dtypes(include=['object']).columns)
    st.subheader(f"Pie Chart de {pie_var}")
    pie_data = df[pie_var].value_counts().reset_index()
    pie_data.columns = [pie_var, 'count']
    if len(pie_data) > 10:
        pie_data = pie_data.head(10)
    pie_chart = px.pie(pie_data, names=pie_var, values='count', title=f'Distribución por {pie_var}', color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(pie_chart)

elif selected == "Correlación y Boxplot":
    st.title("Evaluación de Datos - Correlación y Boxplot")
    st.header("Correlación y Boxplot")
    st.subheader("Mapa de Calor de Correlaciones")
    corr_matrix = numeric_df.corr()
    heatmap_figure = px.imshow(corr_matrix, title="Matriz de Correlación", color_continuous_scale='Viridis', width=800, height=800)
    st.plotly_chart(heatmap_figure)
    boxplot_var = st.sidebar.selectbox("Selecciona una variable para el Boxplot", options=df.select_dtypes(include=['object']).columns)
    st.subheader(f"Boxplot de Precio por {boxplot_var}")
    boxplot_figure = px.box(df, x=boxplot_var, y='price', title=f'Boxplot de Precio por {boxplot_var}', color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(boxplot_figure)

elif selected == "Gráfico de Dispersión":
    st.title("Evaluación de Datos - Gráfico de Dispersión")
    st.header("Gráfico de Dispersión")
    x_var = st.sidebar.selectbox("Selecciona la variable X", options=numeric_cols)
    y_var = st.sidebar.selectbox("Selecciona la variable Y", options=numeric_cols)
    st.subheader(f"Gráfico de Dispersión: {x_var} vs {y_var}")
    scatter_figure = px.scatter(df, x=x_var, y=y_var, title=f'Dispersión: {x_var} vs {y_var}', color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(scatter_figure)

elif selected == "Modelos de Regresión":
    st.title("Evaluación de Datos - Modelos de Regresión")
    st.header("Regresiones")
    regression_type = st.sidebar.selectbox("Selecciona el tipo de regresión", options=["Regresión Lineal Simple", "Regresión Lineal Múltiple", "Regresión No Lineal", "Regresión Logística"])
    x_var = st.sidebar.selectbox("Selecciona la variable X", options=numeric_cols)
    y_var = st.sidebar.selectbox("Selecciona la variable Y", options=numeric_cols)
    X = df[[x_var]].dropna()
    y = df[y_var].dropna()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    if regression_type == "Regresión Lineal Simple":
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r_squared = model.score(X_test, y_test)
        fig = px.scatter(x=X_test[x_var], y=y_test.values.flatten(), title=f'Regresión Lineal Simple: {x_var} vs {y_var}', color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.add_trace(go.Scatter(x=X_test[x_var], y=y_pred, mode='lines', name='Predicción', line=dict(color='red')))
        st.plotly_chart(fig)
        st.write(f"R²: {r_squared}")
        st.write(f"MSE: {mse}")

    elif regression_type == "Regresión Lineal Múltiple":
        x_vars_mult = st.sidebar.multiselect("Selecciona las variables X para la regresión múltiple", options=numeric_cols)
        if len(x_vars_mult) > 1:
            X_mult = df[x_vars_mult].dropna()
            X_train_mult, X_test_mult, y_train_mult, y_test_mult = train_test_split(X_mult, y, test_size=0.3, random_state=42)
            model_mult = LinearRegression()
            model_mult.fit(X_train_mult, y_train_mult)
            y_pred_mult = model_mult.predict(X_test_mult)
            mse_mult = mean_squared_error(y_test_mult, y_pred_mult)
            r_squared_mult = model_mult.score(X_test_mult, y_test_mult)
            fig = px.scatter(x=X_test_mult[x_vars_mult[0]], y=y_test_mult.values.flatten(), title=f'Regresión Lineal Múltiple: {x_vars_mult} vs {y_var}', color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.add_trace(go.Scatter(x=X_test_mult[x_vars_mult[0]], y=y_pred_mult, mode='lines', name='Predicción', line=dict(color='red')))
            st.plotly_chart(fig)
            st.write(f"R²: {r_squared_mult}")
            st.write(f"MSE: {mse_mult}")
        else:
            st.write("Selecciona más de una variable para la regresión múltiple.")

    elif regression_type == "Regresión No Lineal":
        degree = st.sidebar.slider("Selecciona el grado del polinomio", min_value=2, max_value=5, value=2)
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X_train)
        model_poly = LinearRegression()
        model_poly.fit(X_poly, y_train)
        X_test_poly = poly.transform(X_test)
        y_pred_poly = model_poly.predict(X_test_poly)
        fig = px.scatter(x=X_test[x_var], y=y_test.values.flatten(), title=f'Regresión No Lineal: {x_var} vs {y_var}', color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.add_trace(go.Scatter(x=X_test[x_var], y=y_pred_poly, mode='lines', name='Predicción', line=dict(color='red')))
        st.plotly_chart(fig)
        st.write(f"R²: {model_poly.score(X_test_poly, y_test)}")

    elif regression_type == "Regresión Logística":
        binary_vars = ['host_is_superhost', 'instant_bookable', 'has_profile_pic', 'has_identity_verified']
        logistic_var = st.sidebar.selectbox("Selecciona la variable binaria", options=binary_vars)
        X_log = df[[x_var]].dropna()
        y_log = df[logistic_var].dropna()
        X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_log, y_log, test_size=0.3, random_state=42)
        logistic_model = LogisticRegression()
        logistic_model.fit(X_train_log, y_train_log)
        y_pred_log = logistic_model.predict(X_test_log)
        accuracy = accuracy_score(y_test_log, y_pred_log)
        fig = px.scatter(x=X_test_log[x_var], y=y_test_log.values.flatten(), title=f'Regresión Logística: {x_var} vs {logistic_var}', color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.add_trace(go.Scatter(x=X_test_log[x_var], y=y_pred_log, mode='lines', name='Predicción', line=dict(color='red')))
        st.plotly_chart(fig)
        st.write(f"Precisión del modelo logístico: {accuracy}")
