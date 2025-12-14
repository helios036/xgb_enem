"""
Módulo de treinamento de modelo XGBoost para o projeto ML ENEM.
"""

import sys
import os

# Adicionar o diretório atual ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from .importacao_pacotes import importar_pacotes
    from .importacao_dados import preparar_dados_treino_teste
except ImportError:
    from importacao_pacotes import importar_pacotes
    from importacao_dados import preparar_dados_treino_teste

def treinar_xgboost(
    X_train, 
    X_test, 
    y_train, 
    y_test,
    param_grid=None,
    n_splits=50,
    n_iter=20,
    random_state=42,
    n_jobs=-1,
    verbose=True
):
    """
    Treina um modelo XGBoost com otimização de hiperparâmetros usando RandomizedSearchCV.
    
    Args:
        X_train: Features de treino
        X_test: Features de teste
        y_train: Target de treino
        y_test: Target de teste
        param_grid (dict, optional): Grade de parâmetros para busca. Se None, usa padrão.
        n_splits (int): Número de folds para validação cruzada. Padrão: 50
        n_iter (int): Número de iterações da busca aleatória. Padrão: 20
        random_state (int): Seed para reprodutibilidade. Padrão: 42
        n_jobs (int): Número de jobs paralelos. Padrão: -1 (todos os cores)
        verbose (bool): Se True, imprime progresso e resultados. Padrão: True
    
    Returns:
        dict: Dicionário contendo:
            - 'modelo': Melhor modelo treinado
            - 'metricas': DataFrame com métricas de avaliação
            - 'melhores_parametros': Melhores parâmetros encontrados
            - 'y_pred': Previsões no conjunto de teste
            - 'random_search': Objeto RandomizedSearchCV completo
    
    Example:
        >>> resultado = treinar_xgboost(X_train, X_test, y_train, y_test)
        >>> modelo = resultado['modelo']
        >>> metricas = resultado['metricas']
        >>> print(metricas)
    """
    # Importar pacotes necessários
    pacotes = importar_pacotes()
    np = pacotes['np']
    pd = pacotes['pd']
    xgb = pacotes['xgb']
    KFold = pacotes['KFold']
    RandomizedSearchCV = pacotes['RandomizedSearchCV']
    make_scorer = pacotes['make_scorer']
    mean_absolute_percentage_error = pacotes['mean_absolute_percentage_error']
    r2_score = pacotes['r2_score']
    mean_squared_error = pacotes['mean_squared_error']
    mean_absolute_error = pacotes['mean_absolute_error']
    
    if verbose:
        print(f"\n{'='*70}")
        print("TREINAMENTO DE MODELO XGBOOST")
        print(f"{'='*70}")
    
    # Converter para float32 se necessário
    if not all(X_train.dtypes == 'float32'):
        if verbose:
            print("Convertendo dados para float32...")
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
    
    # Definir grade de parâmetros padrão se não fornecida
    if param_grid is None:
        param_grid = {
            'max_depth': np.arange(1, 21, step=2),
            'learning_rate': np.arange(0.01, 0.08, step=0.02),
            'n_estimators': np.arange(20, 100, step=10),
            'subsample': np.arange(0.1, 0.8, step=0.2),
            'colsample_bytree': np.arange(0.1, 0.7, step=0.2),
            'gamma': np.arange(0.1, 0.6, step=0.2),
            'reg_lambda': np.arange(0.1, 1, step=0.2),
            'min_child_weight': np.arange(1, 10, step=2)
        }
    
    if verbose:
        print(f"\nConfiguração:")
        print(f"  - Validação cruzada: {n_splits} folds")
        print(f"  - Iterações de busca: {n_iter}")
        print(f"  - Jobs paralelos: {n_jobs}")
        print(f"  - Shape treino: {X_train.shape}")
        print(f"  - Shape teste: {X_test.shape}")
        print(f"\nIniciando busca de hiperparâmetros...")
    
    # Validação cruzada
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Scorer (MAPE - quanto menor, melhor)
    scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)
    
    # Modelo base
    xgb_model = xgb.XGBRegressor(random_state=random_state)
    
    # RandomizedSearchCV
    random_search_xgb = RandomizedSearchCV(
        xgb_model, 
        param_distributions=param_grid, 
        cv=kf, 
        scoring=scorer, 
        n_jobs=n_jobs, 
        n_iter=n_iter, 
        random_state=random_state,
        verbose=1 if verbose else 0
    )
    
    # Treinar
    random_search_xgb.fit(X_train, y_train)
    
    # Melhor modelo encontrado
    melhor_modelo_xgb = random_search_xgb.best_estimator_
    
    if verbose:
        print(f"\n✓ Treinamento concluído!")
        print(f"\nMelhores parâmetros encontrados:")
        for param, valor in random_search_xgb.best_params_.items():
            print(f"  - {param}: {valor}")
    
    # Previsão
    if verbose:
        print(f"\nRealizando previsões no conjunto de teste...")
    y_pred = melhor_modelo_xgb.predict(X_test)
    
    # Avaliação
    r2_xgb = r2_score(y_test, y_pred)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred))
    mae_xgb = mean_absolute_error(y_test, y_pred)
    mape_xgb = mean_absolute_percentage_error(y_test, y_pred)
    
    # DataFrame de métricas
    metricas_resultados = pd.DataFrame({
        'Métrica': ['R²', 'RMSE', 'MAE', 'MAPE'],
        'Valor': [r2_xgb, rmse_xgb, mae_xgb, mape_xgb]
    })
    
    if verbose:
        print(f"\n{'='*70}")
        print("MÉTRICAS DE AVALIAÇÃO")
        print(f"{'='*70}")
        print(metricas_resultados.to_string(index=False))
        print(f"{'='*70}\n")
    
    # Retornar resultados
    return {
        'modelo': melhor_modelo_xgb,
        'metricas': metricas_resultados,
        'melhores_parametros': random_search_xgb.best_params_,
        'y_pred': y_pred,
        'random_search': random_search_xgb
    }


def treinar_modelo_completo(
    limite_linhas=None,
    table_name="ENEM_2021_ONEHOT_V3",
    project_id="modelagem-1971",
    dataset_id="ENEM",
    location="us",
    param_grid=None,
    n_splits=50,
    n_iter=20,
    random_state=42,
    verbose=True
):
    """
    Pipeline completo: importa dados do BigQuery e treina modelo XGBoost.
    
    Args:
        limite_linhas (int, optional): Limita o número de linhas importadas do BigQuery.
                                       Use para testes rápidos (ex: 10000).
        table_name (str): Nome da tabela no BigQuery.
        project_id (str): ID do projeto no Google Cloud.
        dataset_id (str): ID do dataset no BigQuery.
        location (str): Localização do dataset.
        param_grid (dict, optional): Grade de parâmetros para busca de hiperparâmetros.
        n_splits (int): Número de folds para validação cruzada.
        n_iter (int): Número de iterações da busca aleatória.
        random_state (int): Seed para reprodutibilidade.
        verbose (bool): Se True, imprime progresso detalhado.
    
    Returns:
        dict: Dicionário contendo:
            - 'modelo': Melhor modelo treinado
            - 'metricas': DataFrame com métricas
            - 'melhores_parametros': Melhores hiperparâmetros
            - 'y_pred': Previsões no teste
            - 'y_test': Valores reais do teste
            - 'X_train': Features de treino
            - 'X_test': Features de teste
            - 'random_search': Objeto RandomizedSearchCV
    
    Examples:
        >>> # Teste rápido com 10 mil linhas
        >>> resultado = treinar_modelo_completo(limite_linhas=10000)
        >>> 
        >>> # Treinamento completo
        >>> resultado = treinar_modelo_completo()
        >>> 
        >>> # Acessar resultados
        >>> modelo = resultado['modelo']
        >>> metricas = resultado['metricas']
        >>> print(metricas)
    """
    if verbose:
        print(f"\n{'#'*70}")
        print("PIPELINE COMPLETO DE TREINAMENTO XGBoost")
        print(f"{'#'*70}\n")
    
    # Etapa 1: Importar e preparar dados
    if verbose:
        print("ETAPA 1: Importação e preparação dos dados")
    
    X_train, X_test, y_train, y_test = preparar_dados_treino_teste(
        table_name=table_name,
        project_id=project_id,
        dataset_id=dataset_id,
        location=location,
        limite_linhas=limite_linhas,
        random_state=random_state
    )
    
    # Etapa 2: Treinar modelo
    if verbose:
        print("\nETAPA 2: Treinamento do modelo XGBoost")
    
    resultado = treinar_xgboost(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        param_grid=param_grid,
        n_splits=n_splits,
        n_iter=n_iter,
        random_state=random_state,
        verbose=verbose
    )
    
    # Adicionar dados ao resultado
    resultado['y_test'] = y_test
    resultado['X_train'] = X_train
    resultado['X_test'] = X_test
    
    if verbose:
        print(f"\n{'#'*70}")
        print("✓ PIPELINE CONCLUÍDO COM SUCESSO!")
        print(f"{'#'*70}\n")
    
    return resultado


if __name__ == "__main__":
    # Exemplo de uso completo
    print("="*70)
    print("EXECUTANDO PIPELINE COMPLETO DE TREINAMENTO")
    print("="*70)
    
    # Treinar com amostra de 5000 linhas (rápido para teste)
    resultado = treinar_modelo_completo(limite_linhas=5000, n_iter=5)
    
    print("\n" + "="*70)
    print("RESULTADOS FINAIS:")
    print("="*70)
    print(resultado['metricas'])
    print("\nModelo treinado e pronto para uso!")
    print("="*70)