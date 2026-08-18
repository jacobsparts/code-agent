import pytest


@pytest.fixture(autouse=True)
def isolated_provider_admission(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "CODE_AGENT_ADMISSION_DB", str(tmp_path / "provider_admission.sqlite3")
    )
    monkeypatch.setenv(
        "CODE_AGENT_ADMISSION_NOTIFY", str(tmp_path / "provider_admission.notify")
    )
