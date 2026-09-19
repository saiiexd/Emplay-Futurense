import os
import unittest
from unittest.mock import MagicMock, patch
from src.extraction.llm_client import MistralChatProvider, ProviderConfigError, ProviderCallError

class TestMistralProvider(unittest.TestCase):

    @patch("os.getenv")
    def test_missing_api_key(self, mock_getenv):
        mock_getenv.return_value = ""
        with self.assertRaises(ProviderConfigError) as ctx:
            MistralChatProvider()
        self.assertIn("MISTRAL_API_KEY is not set", str(ctx.exception))

    @patch("mistralai.client.Mistral")
    @patch("os.getenv")
    def test_successful_structured_output(self, mock_getenv, mock_mistral):
        mock_getenv.return_value = "dummy-key"
        
        # Mock the client response
        mock_client_instance = MagicMock()
        mock_mistral.return_value = mock_client_instance
        
        mock_choice = MagicMock()
        mock_choice.message.content = '{"field": "value"}'
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client_instance.chat.complete.return_value = mock_response

        provider = MistralChatProvider(api_key="dummy-key", model="mistral-small-latest")
        result = provider.complete(
            messages=[{"role": "user", "content": "hi"}],
            max_output_tokens=100,
            response_schema={"type": "object"}
        )
        
        self.assertEqual(result, '{"field": "value"}')
        mock_client_instance.chat.complete.assert_called_once()
        
        kwargs = mock_client_instance.chat.complete.call_args[1]
        self.assertEqual(kwargs["model"], "mistral-small-latest")
        self.assertEqual(kwargs["max_tokens"], 100)
        self.assertEqual(kwargs["response_format"], {"type": "json_object"})

    @patch("mistralai.client.Mistral")
    def test_authentication_error(self, mock_mistral):
        mock_client = MagicMock()
        mock_mistral.return_value = mock_client
        
        # Simulate an exception with status_code 401
        class FakeMistralException(Exception):
            status_code = 401
            
        mock_client.chat.complete.side_effect = FakeMistralException("Unauthorized")
        
        provider = MistralChatProvider(api_key="dummy")
        with self.assertRaises(ProviderConfigError) as ctx:
            provider.complete([{"role": "user", "content": "hi"}], max_output_tokens=100)
            
        self.assertIn("authentication rejected", str(ctx.exception))

if __name__ == "__main__":
    unittest.main()
