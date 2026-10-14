import pytest

from docrag.config import Settings
from tests.fakes import make_settings


@pytest.fixture
def settings() -> Settings:
    """Default settings, isolated from any local .env file."""
    return make_settings()
