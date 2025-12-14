import sys
import os

# Adicionar o diretório atual ao path para permitir importações
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from .importacao_pacotes import importar_pacotes
except ImportError:
    from importacao_pacotes import importar_pacotes


def importar_dados_enem(
    table_name="ENEM_2021_ONEHOT_V3",
    project_id="modelagem-1971",
    dataset_id="ENEM",
    location="us",
    sql_custom=None
):
    # Importar pacotes necessários
    pacotes = importar_pacotes()
    bigquery = pacotes['bigquery']
    
    # Construir o caminho da tabela
    table_path = f"{project_id}.{dataset_id}.{table_name}"
    
    # Criar cliente do BigQuery
    print("Conectando ao BigQuery...")
    client = bigquery.Client(project=project_id, location=location)
    
    # Definir a query SQL
    if sql_custom is None:
        sql_query = f"SELECT * FROM `{table_path}`"
    else:
        sql_query = sql_custom
    
    # Executar a query e retornar o DataFrame
    print(f"Executando query: {sql_query[:100]}...")
    print("Aguardando resposta do BigQuery (isso pode demorar alguns minutos)...")
    
    query_job = client.query(sql_query)
    dataset_enem = query_job.to_dataframe()
    
    print(f"✓ Dados importados com sucesso: {dataset_enem.shape[0]} linhas e {dataset_enem.shape[1]} colunas")
    
    return dataset_enem


def preparar_dados_treino_teste(
    table_name="amostra_1",
    project_id="modelagem-1971",
    dataset_id="dataform_deploy",
    location="us",
    sql_custom=None,
    limite_linhas=None,
    target_column='media_final',
    test_size=0.20,
    random_state=42
):
    """
    Importa dados do BigQuery e prepara conjuntos de treino e teste.
    
    Args:
        table_name (str): Nome da tabela no BigQuery.
        project_id (str): ID do projeto no Google Cloud.
        dataset_id (str): ID do dataset no BigQuery.
        location (str): Localização do dataset.
        sql_custom (str, optional): Query SQL customizada.
        limite_linhas (int, optional): Limita o número de linhas importadas (acelera o processo).
        target_column (str): Nome da coluna target. Padrão: 'media_final'
        test_size (float): Proporção do conjunto de teste. Padrão: 0.20
        random_state (int): Seed para reprodutibilidade. Padrão: 42
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test) - Todos em formato float32
    
    Examples:
        >>> # Importar amostra de 10000 linhas (rápido para testes)
        >>> X_train, X_test, y_train, y_test = preparar_dados_treino_teste(limite_linhas=10000)
        
        >>> # Importar todos os dados
        >>> X_train, X_test, y_train, y_test = preparar_dados_treino_teste()
    """
    # Importar pacotes necessários
    pacotes = importar_pacotes()
    train_test_split = pacotes['train_test_split']
    
    # Construir query com limite se especificado
    if sql_custom is None and limite_linhas is not None:
        table_path = f"{project_id}.{dataset_id}.{table_name}"
        sql_custom = f"SELECT * FROM `{table_path}` LIMIT {limite_linhas}"
    
    # Importar dados
    print(f"\n{'='*60}")
    print("PREPARANDO DADOS DE TREINO E TESTE")
    print(f"{'='*60}")
    dataset_enem = importar_dados_enem(
        table_name=table_name,
        project_id=project_id,
        dataset_id=dataset_id,
        location=location,
        sql_custom=sql_custom
    )
    
    # Limpeza de dados
    print(f"\nLimpando dados...")
    linhas_iniciais = len(dataset_enem)
    
    # Remover linhas com NaN na coluna target
    dataset_enem = dataset_enem.dropna(subset=[target_column])
    
    # Remover linhas com infinito na coluna target
    pacotes = importar_pacotes()
    np = pacotes['np']
    dataset_enem = dataset_enem[np.isfinite(dataset_enem[target_column])]
    
    # Remover linhas com NaN em qualquer feature
    dataset_enem = dataset_enem.dropna()
    
    # Remover infinitos em todas as colunas
    dataset_enem = dataset_enem[np.isfinite(dataset_enem).all(axis=1)]
    
    linhas_removidas = linhas_iniciais - len(dataset_enem)
    if linhas_removidas > 0:
        print(f"⚠ {linhas_removidas} linhas removidas (NaN ou infinito)")
    print(f"✓ Dataset limpo: {len(dataset_enem)} linhas válidas")
    
    # Separar features e target
    print(f"\nSeparando features e target (coluna: {target_column})...")
    dataset_enem_features = dataset_enem.drop(columns=[target_column])
    dataset_enem_target = dataset_enem[target_column]
    
    # Dividir em treino e teste
    print(f"Dividindo dados: {int((1-test_size)*100)}% treino, {int(test_size*100)}% teste...")
    X_train, X_test, y_train, y_test = train_test_split(
        dataset_enem_features, 
        dataset_enem_target, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # Converter para float32 (economiza memória e acelera treinamento)
    print("Convertendo para float32...")
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    
    print(f"\n{'='*60}")
    print("✓ DADOS PREPARADOS COM SUCESSO!")
    print(f"{'='*60}")
    print(f"Treino: {X_train.shape[0]} linhas x {X_train.shape[1]} features")
    print(f"Teste:  {X_test.shape[0]} linhas x {X_test.shape[1]} features")
    print(f"{'='*60}\n")
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Exemplo de uso: importar amostra de 5000 linhas (rápido para teste)
    X_train, X_test, y_train, y_test = preparar_dados_treino_teste(limite_linhas=5000)
    print(f"\nPrimeiras linhas do conjunto de treino:")
    print(X_train.head())