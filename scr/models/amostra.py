import sys
import os

# Adicionar o diretório atual ao path para permitir importações
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from .importacao_dados import preparar_dados_treino_teste
except ImportError:
    from importacao_dados import preparar_dados_treino_teste

if __name__ == "__main__":
    # Opção 1: Testar com amostra pequena (rápido - 10 mil linhas)
    print("Preparando amostra de 10.000 linhas para teste rápido...")
    X_train, X_test, y_train, y_test = preparar_dados_treino_teste(limite_linhas=500000)
    
    # Opção 2: Para usar todos os dados, descomente a linha abaixo:
    # X_train, X_test, y_train, y_test = preparar_dados_treino_teste()
    
    print("\nResumo dos dados:")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")