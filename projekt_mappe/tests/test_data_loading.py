# tests/test_data_loading.py
import unittest
import pandas as pd
from modules.data_loading import load_data

class TestDataLoading(unittest.TestCase):
    def test_load_data(self):
        # Brug testfiler eller mock data
        covariance_matrix, init_values = load_data("data/covariance_matrix.xlsx", "data/init_values.xlsx")
        
        # Test om dataframes ikke er tomme
        self.assertFalse(covariance_matrix.empty, "Covariance matrix er tom")
        self.assertFalse(init_values.empty, "Init values er tomme")
        
        # Test om dimensioner er korrekte
        self.assertEqual(covariance_matrix.shape[0], covariance_matrix.shape[1], "Covariance matrix er ikke kvadratisk")
        self.assertEqual(len(init_values), covariance_matrix.shape[0], "Init values og covariance matrix har ikke samme antal aktiver")

if __name__ == '__main__':
    unittest.main()
