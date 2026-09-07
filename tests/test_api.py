import asyncio
import io

import pytest
from fastapi import HTTPException, UploadFile
from starlette.datastructures import Headers

from crop_copilot.api import main
from crop_copilot.config import Settings


def upload(payload):
    return UploadFile(
        filename="leaf.png", file=io.BytesIO(payload),
        headers=Headers({"content-type": "image/png"}),
    )


@pytest.mark.parametrize("payload,status", [(b"", 422), (b"12345", 413)])
def test_invalid_upload_cleans_temporary_file(tmp_path, monkeypatch, payload, status):
    monkeypatch.setattr(main.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 4)
    with pytest.raises(HTTPException) as error:
        asyncio.run(main.save_limited_upload(upload(payload), main.ALLOWED_IMAGE_TYPES))
    assert error.value.status_code == status
    assert list(tmp_path.iterdir()) == []


def test_default_reviewer_key_is_disabled_in_production(monkeypatch):
    settings = Settings(_env_file=None, environment="production")
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    with pytest.raises(HTTPException) as error:
        main.verify_reviewer_key("dev-review-key")
    assert error.value.status_code == 503


def test_wrong_unicode_reviewer_key_returns_unauthorized(monkeypatch):
    settings = Settings(_env_file=None, reviewer_api_key="test-private-key")
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    with pytest.raises(HTTPException) as error:
        main.verify_reviewer_key("ಕನ್ನಡ")
    assert error.value.status_code == 401
