# tests/test_data_analysis.py

import pandas as pd
import numpy as np
import pytest
from src.data_analysis import clean_data, compute_correlation, perform_pca

@pytest.fixture
def sample_data():
    data = {
        'A': [1, 2, 3, np.nan, 5, 1000],
        'B': [5, np.nan, 6, 7, 8, 9],
        'C': [10, 11, 12, 13, 14, 15]
    }
    df = pd.DataFrame(data)
    return df

def test_clean_data(sample_data):
    cleaned_df = clean_data(sample_data)
    assert cleaned_df.isna().sum().sum() == 0  # Verifica se não há valores faltantes
    assert 1000 not in cleaned_df['A'].values  # Verifica se o outlier foi removido

def test_compute_correlation(sample_data):
    sample_data = sample_data.dropna()
    correlation_matrix = compute_correlation(sample_data)
    assert correlation_matrix.shape == (3, 3)  # Verifica o tamanho da matriz de correlação
    assert np.isclose(correlation_matrix.loc['A', 'B'], 0.5, atol=0.1)  # Verifica se a correlação é aproximadamente 0.5

def test_perform_pca(sample_data):
    sample_data = sample_data.dropna()
    principal_components = perform_pca(sample_data)
    assert principal_components.shape[1] == 2  # Verifica se o número de componentes principais é 2
    assert principal_components.shape[0] == len(sample_data)  # Verifica o número de observações
