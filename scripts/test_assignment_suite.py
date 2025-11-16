"""Unit тесты для всех 10 заданий
Все тесты импортируют функции из assignment.py
"""

# test_01_iris.py
import unittest
import pandas as pd
from pathlib import Path

class TestIrisAssignment(unittest.TestCase):
    def test_load_data_shape(self):
        try:
            from assignment import load_data
            df = load_data()
            self.assertEqual(df.shape, (150, 5))
        except ImportError:
            self.skipTest("assignment.py не найден")
    
    def test_visualize_creates_files(self):
        try:
            from assignment import load_data, visualize_target
            df = load_data()
            visualize_target(df)
            self.assertGreater(len(list(Path('.').glob('*.png'))), 0)
        except ImportError:
            self.skipTest("assignment.py не найден")


# test_02_wine.py
class TestWineAssignment(unittest.TestCase):
    def test_load_data_shape(self):
        try:
            from assignment import load_data
            df = load_data()
            self.assertEqual(df.shape, (178, 14))
        except ImportError:
            self.skipTest("assignment.py не найден")
    
    def test_target_classes(self):
        try:
            from assignment import load_data
            df = load_data()
            self.assertEqual(df.iloc[:, -1].nunique(), 3)
        except ImportError:
            self.skipTest("assignment.py не найден")


# test_03_diabetes.py
class TestDiabetesAssignment(unittest.TestCase):
    def test_load_data_shape(self):
        try:
            from assignment import load_data
            df = load_data()
            self.assertEqual(df.shape, (442, 11))
        except ImportError:
            self.skipTest("assignment.py не найден")
    
    def test_target_numeric(self):
        try:
            from assignment import load_data
            df = load_data()
            self.assertTrue(pd.api.types.is_numeric_dtype(df.iloc[:, -1]))
        except ImportError:
            self.skipTest("assignment.py не найден")


# test_04_breast_cancer.py
class TestBreastCancerAssignment(unittest.TestCase):
    def test_load_data_shape(self):
        try:
            from assignment import load_data
            df = load_data()
            self.assertEqual(df.shape, (569, 31))
        except ImportError:
            self.skipTest("assignment.py не найден")
    
    def test_binary_classification(self):
        try:
            from assignment import load_data
            df = load_data()
            self.assertEqual(df.iloc[:, -1].nunique(), 2)
        except ImportError:
            self.skipTest("assignment.py не найден")


# test_05_california_housing.py
class TestHousingAssignment(unittest.TestCase):
    def test_load_data_shape(self):
        try:
            from assignment import load_data
            df = load_data()
            self.assertEqual(df.shape, (20640, 9))
        except ImportError:
            self.skipTest("assignment.py не найден")


# test_06_titanic.py
class TestTitanicAssignment(unittest.TestCase):
    def test_load_data_reasonable_size(self):
        try:
            from assignment import load_data
            df = load_data()
            self.assertGreater(len(df), 800)
            self.assertLess(len(df), 950)
        except ImportError:
            self.skipTest("assignment.py не найден")


# test_07_german_credit.py
class TestGermanCreditAssignment(unittest.TestCase):
    def test_load_data_size(self):
        try:
            from assignment import load_data
            df = load_data()
            self.assertEqual(len(df), 1000)
        except ImportError:
            self.skipTest("assignment.py не найден")


# test_08_wine_quality.py
class TestWineQualityAssignment(unittest.TestCase):
    def test_load_data_large(self):
        try:
            from assignment import load_data
            df = load_data()
            self.assertGreater(len(df), 5000)
        except ImportError:
            self.skipTest("assignment.py не найден")


# test_09_heart_disease.py
class TestHeartDiseaseAssignment(unittest.TestCase):
    def test_load_data_loaded(self):
        try:
            from assignment import load_data
            df = load_data()
            self.assertGreater(len(df), 100)
        except ImportError:
            self.skipTest("assignment.py не найден")


# test_10_digits.py
class TestDigitsAssignment(unittest.TestCase):
    def test_load_data_shape(self):
        try:
            from assignment import load_data
            df = load_data()
            self.assertEqual(df.shape, (1797, 65))
        except ImportError:
            self.skipTest("assignment.py не найден")
    
    def test_multiclass_target(self):
        try:
            from assignment import load_data
            df = load_data()
            self.assertEqual(df.iloc[:, -1].nunique(), 10)
        except ImportError:
            self.skipTest("assignment.py не найден")


if __name__ == '__main__':
    unittest.main()
