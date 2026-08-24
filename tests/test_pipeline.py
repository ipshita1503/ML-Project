"""Tests for the GIIP pipeline."""
import os
import sys
import unittest
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data.loader import load_vaccine_data, validate_data
from src.features.engineer import engineer_features

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.df_raw = load_vaccine_data()

    def test_data_loading(self):
        self.assertFalse(self.df_raw.empty)
        self.assertIn('MCV1_TARGET', self.df_raw.columns)
        
    def test_validation(self):
        val = validate_data(self.df_raw)
        self.assertEqual(val['duplicates'], 0)
        self.assertEqual(len(val['countries']), 3)

    def test_feature_engineering(self):
        df_eng, features = engineer_features(self.df_raw)
        self.assertFalse(df_eng.empty)
        self.assertTrue(len(features) > 0)
        self.assertIn('MCV1_TARGET', df_eng.columns)

if __name__ == '__main__':
    unittest.main()
