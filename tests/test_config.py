import pytest
from pydantic import ValidationError

from crop_copilot.config import Settings


def test_demo_classifier_is_blocked_in_production() -> None:
    with pytest.raises(ValidationError, match="cannot be enabled in production"):
        Settings(environment="production", allow_demo_classifier=True)


def test_demo_classifier_is_allowed_for_local_development() -> None:
    settings = Settings(environment="development", allow_demo_classifier=True)
    assert settings.allow_demo_classifier is True

