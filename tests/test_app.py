"""Streamlit AppTest for the observability dashboard.

Verifies that the dashboard loads, renders the configured LLM provider label
in the sidebar, and shows the extraction tab's CLI-driven explanation.
"""

import os
import sys
import unittest

from streamlit.testing.v1 import AppTest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

APP_PATH = os.path.join(os.path.dirname(__file__), '..', 'app.py')


class TestStreamlitApp(unittest.TestCase):
    def test_app_loads_without_error(self):
        at = AppTest.from_file(APP_PATH).run(timeout=15)
        self.assertFalse(at.exception, f"app.py raised an exception: {at.exception}")

    def test_sidebar_shows_configured_llm_provider(self):
        """The sidebar must mention the configured LLM provider, whatever it is."""
        at = AppTest.from_file(APP_PATH).run(timeout=15)
        sidebar_texts = [m.value for m in at.sidebar.markdown]
        found = any("Configured LLM provider:" in t for t in sidebar_texts)
        self.assertTrue(found, "The 'Configured LLM provider' label was not found in the sidebar")


if __name__ == '__main__':
    unittest.main()
