"""Settings, and the kill switch's parsing rules.

The kill switch is an incident lever: someone types it while a user's API key is
burning money. So the tests below pin the permissive reading -- anything that is
not an explicit "off" turns the switch ON -- because the failure mode of the
strict reading is that `JFL_DISABLE_MODEL_CALLS=yes` keeps spending.
"""

from __future__ import annotations

import base64
import datetime as dt
import os

import pytest
from jfl_core.crypto.envelope import MasterKeyError
from jfl_core.db.tables import LOCAL_USER_ID
from jfl_worker.settings import DEFAULT_MODEL, WorkerSettings, model_calls_disabled

# A throwaway KEK, base64, as the environment would carry it. Generated rather
# than hard-coded so nothing here looks like a key anyone should keep.
_MASTER_KEY = base64.b64encode(os.urandom(32)).decode()


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", "please", " 1 "])
def test_kill_switch_is_on_for_anything_but_an_explicit_off(value: str) -> None:
    assert model_calls_disabled({"JFL_DISABLE_MODEL_CALLS": value}) is True


@pytest.mark.parametrize("value", ["", "0", "false", "FALSE", "no", "off", "  "])
def test_kill_switch_is_off_for_explicit_off_values(value: str) -> None:
    assert model_calls_disabled({"JFL_DISABLE_MODEL_CALLS": value}) is False


def test_kill_switch_is_off_when_unset() -> None:
    assert model_calls_disabled({}) is False


def test_kill_switch_reads_the_environment_at_call_time() -> None:
    """Not captured at startup -- the brief requires a dispatch-time check."""
    env: dict[str, str] = {}
    assert model_calls_disabled(env) is False
    env["JFL_DISABLE_MODEL_CALLS"] = "1"
    assert model_calls_disabled(env) is True


def test_from_env_requires_a_database_url() -> None:
    with pytest.raises(KeyError):
        WorkerSettings.from_env({})


def test_from_env_takes_defaults_and_overrides() -> None:
    settings = WorkerSettings.from_env(
        {
            "JFL_DATABASE_URL": "postgresql+psycopg://x/y",
            "JFL_WORKER_POLL_INTERVAL": "0.5",
            "JFL_MASTER_KEY": _MASTER_KEY,
        }
    )
    assert settings.database_url == "postgresql+psycopg://x/y"
    assert settings.poll_interval == 0.5
    assert settings.visibility_timeout == 15 * 60.0
    assert settings.batch_size == 1
    assert settings.system_user_id == LOCAL_USER_ID
    # No JFL_MODEL set, so the default. Model choice is a product option that a
    # deployment picks, but there is always an answer.
    assert settings.model == DEFAULT_MODEL


def test_the_model_is_read_from_the_environment() -> None:
    settings = WorkerSettings.from_env(
        {
            "JFL_DATABASE_URL": "postgresql+psycopg://x/y",
            "JFL_MASTER_KEY": _MASTER_KEY,
            "JFL_MODEL": "claude-sonnet-4-5",
        }
    )
    assert settings.model == "claude-sonnet-4-5"


def test_a_missing_master_key_fails_at_boot_not_at_the_first_extraction() -> None:
    """The worker now unseals users' API keys, so `JFL_MASTER_KEY` is as
    required as the database URL. Discovering it is absent one task into
    someone's first extraction would mean a failed application and a message
    about credentials that is really a message about deployment.
    """
    with pytest.raises(MasterKeyError):
        WorkerSettings.from_env({"JFL_DATABASE_URL": "postgresql+psycopg://x/y"})


def test_backoff_is_exponential_and_capped() -> None:
    settings = WorkerSettings(database_url="x", retry_base=30.0, retry_factor=2.0, retry_cap=600.0)
    # `attempts` is the count AFTER the failed attempt, so the first failure is 1.
    assert settings.retry_delay(1) == dt.timedelta(seconds=30)
    assert settings.retry_delay(2) == dt.timedelta(seconds=60)
    assert settings.retry_delay(3) == dt.timedelta(seconds=120)
    assert settings.retry_delay(10) == dt.timedelta(seconds=600)  # capped, not 15360s


def test_backoff_never_goes_negative_on_a_zero_attempt_count() -> None:
    settings = WorkerSettings(database_url="x")
    assert settings.retry_delay(0) == dt.timedelta(seconds=30)
