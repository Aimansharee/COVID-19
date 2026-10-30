"""
Tests for visualization components (Plotly and Matplotlib figures).
"""

import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def test_plotly_scatter_actual_vs_predicted():
    """Verify actual vs predicted scatter plot generation."""
    y_test = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([11.0, 19.0, 31.0])
    
    fig = px.scatter(x=y_test, y=y_pred,
                     labels={'x': 'Actual Cases', 'y': 'Predicted Cases'},
                     title='Actual vs Predicted Cases')
    fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()],
                             y=[y_test.min(), y_test.max()],
                             mode='lines', name='Perfect Prediction'))
    
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # scatter trace + line trace
    assert fig.layout.title.text == 'Actual vs Predicted Cases'


def test_plotly_choropleth_generation():
    """Verify choropleth map figure creation with country data."""
    df_geo = pd.DataFrame({
        "location": ["France", "Germany", "Italy"],
        "cases_per_100k": [1200.5, 980.2, 1100.8],
        "total_cases": [50000, 45000, 48000],
        "total_deaths": [1000, 900, 950]
    })
    
    fig = px.choropleth(df_geo,
                        locations="location",
                        locationmode='country names',
                        color="cases_per_100k",
                        hover_name="location",
                        hover_data=["total_cases", "total_deaths"],
                        title="Global COVID-19 Spread")
    
    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "choropleth"


def test_matplotlib_relative_error_figure_cleanup():
    """Verify matplotlib plot can be created, populated, and closed cleanly."""
    y_test_actual = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 305.0])
    relative_errors = (y_pred - y_test_actual) / (y_test_actual + 1e-6)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_test_actual, relative_errors, alpha=0.6, color='green')
    ax.axhline(y=0, color='red', linestyle='--')
    ax.set_title('Relative Error vs Actual')
    
    assert len(ax.collections) == 1  # 1 scatter collection
    assert len(ax.lines) == 1        # 1 horizontal axhline
    
    # Verify cleanup prevents resource leak
    plt.close(fig)
    assert not plt.fignum_exists(fig.number)
