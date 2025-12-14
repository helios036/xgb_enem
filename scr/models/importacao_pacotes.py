"""
Módulo de importação de pacotes para o projeto ML ENEM.
"""

def importar_pacotes():
    """
    Importa e retorna todos os pacotes necessários para o projeto de ML ENEM.
    
    Returns:
        dict: Dicionário contendo todos os módulos e classes importados.
    """
    # Bibliotecas de dados e cloud
    from google.cloud import bigquery
    import pandas as pd
    import numpy as np
    
    # XGBoost
    import xgboost as xgb
    
    # Scikit-learn - Seleção de features
    from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_regression, RFE, RFECV
    
    # Scikit-learn - Pré-processamento
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn import preprocessing
    
    # Scikit-learn - Divisão e validação
    from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit, RandomizedSearchCV
    
    # Scikit-learn - Modelos
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import Pipeline
    
    # Scikit-learn - Métricas
    from sklearn.metrics import (
        mean_squared_error, 
        mean_absolute_error, 
        r2_score,
        make_scorer, 
        mean_absolute_percentage_error
    )
    
    # Visualização
    import matplotlib.pyplot as plt
    from sklearn.tree import plot_tree
    import shap
    
    # Utilitários
    import joblib
    import os
    
    return {
        'bigquery': bigquery,
        'pd': pd,
        'np': np,
        'xgb': xgb,
        'SelectKBest': SelectKBest,
        'f_classif': f_classif,
        'mutual_info_regression': mutual_info_regression,
        'RFE': RFE,
        'RFECV': RFECV,
        'StandardScaler': StandardScaler,
        'MinMaxScaler': MinMaxScaler,
        'preprocessing': preprocessing,
        'train_test_split': train_test_split,
        'KFold': KFold,
        'TimeSeriesSplit': TimeSeriesSplit,
        'RandomizedSearchCV': RandomizedSearchCV,
        'LinearRegression': LinearRegression,
        'RandomForestRegressor': RandomForestRegressor,
        'Pipeline': Pipeline,
        'mean_squared_error': mean_squared_error,
        'mean_absolute_error': mean_absolute_error,
        'r2_score': r2_score,
        'make_scorer': make_scorer,
        'mean_absolute_percentage_error': mean_absolute_percentage_error,
        'plt': plt,
        'plot_tree': plot_tree,
        'shap': shap,
        'joblib': joblib,
        'os': os
    }

