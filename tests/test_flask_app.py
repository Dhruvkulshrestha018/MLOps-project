import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from flask_app.app import app

class FlaskAppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    @patch('flask_app.app.get_model')
    @patch('flask_app.app.get_vectorizer')
    def test_home_page(self, mock_vectorizer, mock_model):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<title>Sentiment Analysis</title>', response.data)

    @patch('flask_app.app.get_model')
    @patch('flask_app.app.get_vectorizer')
    def test_predict_page(self, mock_vectorizer, mock_model):
        # Setup mocks
        mock_vectorizer.return_value.transform.return_value.toarray.return_value = [[0] * 100]
        mock_model.return_value.predict.return_value = np.array([1])  # 1 for Positive
        
        response = self.client.post('/predict', data=dict(text="I love this!"))
        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            b'Positive' in response.data or b'Negative' in response.data,
            "Response should contain either 'Positive' or 'Negative'"
        )

if __name__ == '__main__':
    unittest.main()