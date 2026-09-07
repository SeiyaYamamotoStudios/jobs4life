"""A3's acceptance criterion: tenancy is enforced by construction, and a test
says so rather than a reviewer.

The failure being prevented is a missing `WHERE user_id = ...`, which hands one
person's career history to another. The rule this file enforces is mechanical:

  1. every class named `*Repository` anywhere in `jfl_core` or `jfl_web` must
     declare which kind it is, by subclassing `TenantScopedRepository` or
     `PreAuthRepository`;
  2. no public method of a tenancy-scoped repository may take a parameter whose
     name mentions `user_id` -- the id it may act on is fixed in `__init__` and
     there is no per-call override to forget;
  3. a tenancy-scoped repository must genuinely require that id at construction,
     and must expose no method that swaps it afterwards.

**Discovery is automatic.** Modules are walked with `pkgutil`, not listed, so a
repository written next year is covered without anyone remembering this file
exists. A new repository that subclasses neither base fails rule 1 with a message
saying what to do -- which is the point: the test's job is to make the omission
impossible to ship, not to catch it only where someone thought to look.

The one exemption is `jfl_core.storage.postgres` (with the protocols in
`jfl_core.repositories`): the single-user CLI's storage layer, written before
this scheme and taking `user_id` per call. It is named explicitly, so the
exemption is a decision on the record rather than a gap. New repositories do not
join it -- adding a module to `_LEGACY_MODULES` should feel like what it is.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from types import ModuleType

import pytest
from jfl_core.storage.tenancy import PreAuthRepository, TenantScopedRepository

_PACKAGES = ("jfl_core", "jfl_web")

# Pre-dates structural tenancy. See the module docstring; do not extend.
_LEGACY_MODULES = frozenset(
    {
        "jfl_core.repositories",  # Protocols for the CLI-era repositories
        "jfl_core.storage.postgres",  # their Postgres implementation
    }
)

# The base classes themselves are the scheme, not participants in it.
_BASES = (TenantScopedRepository, PreAuthRepository)

# Anything that would let a caller re-point an instance at another tenant.
_FORBIDDEN_METHOD_NAMES = frozenset({"for_user", "with_user", "set_user", "as_user", "scope_to"})


def _walk_modules() -> list[ModuleType]:
    found: list[ModuleType] = []
    for package_name in _PACKAGES:
        package = importlib.import_module(package_name)
        found.append(package)
        for info in pkgutil.walk_packages(package.__path__, prefix=f"{package_name}."):
            found.append(importlib.import_module(info.name))
    return found


def _repository_classes() -> list[type]:
    """Every `*Repository` class defined (not merely imported) in the packages."""
    classes: dict[str, type] = {}
    for module in _walk_modules():
        for name, obj in vars(module).items():
            if not inspect.isclass(obj) or not name.endswith("Repository"):
                continue
            if obj in _BASES:
                continue
            if obj.__module__ not in {m.__name__ for m in [module]}:
                continue  # imported from elsewhere; it is checked where it lives
            classes[f"{obj.__module__}.{obj.__qualname__}"] = obj
    return [classes[key] for key in sorted(classes)]


def _public_methods(cls: type) -> list[tuple[str, object]]:
    return [
        (name, func)
        for name, func in inspect.getmembers(cls, predicate=inspect.isfunction)
        if not name.startswith("_")
    ]


ALL_REPOSITORIES = _repository_classes()
SCOPED = [c for c in ALL_REPOSITORIES if issubclass(c, TenantScopedRepository)]


def test_discovery_actually_found_something() -> None:
    """A discovery test that silently finds nothing passes and proves nothing."""
    assert SCOPED, "no tenancy-scoped repositories discovered -- is the walk broken?"


@pytest.mark.parametrize("cls", ALL_REPOSITORIES, ids=lambda c: f"{c.__module__}.{c.__name__}")
def test_every_repository_declares_its_kind(cls: type) -> None:
    if cls.__module__ in _LEGACY_MODULES:
        return
    assert issubclass(cls, _BASES), (
        f"{cls.__module__}.{cls.__name__} is a repository but subclasses neither "
        "TenantScopedRepository nor PreAuthRepository. Anything that reads or "
        "writes a user's content must be tenancy-scoped; only identity resolution "
        "and session lookup may be PreAuthRepository, and must say why in its "
        "docstring."
    )


@pytest.mark.parametrize("cls", SCOPED, ids=lambda c: f"{c.__module__}.{c.__name__}")
def test_no_scoped_method_accepts_a_user_id(cls: type) -> None:
    """Rule 2, and the reason this whole file exists.

    If a method can be handed a `user_id`, then somewhere there is a call site
    that has to pass the right one, and a call site that has to pass the right
    one is a call site that can pass the wrong one.
    """
    offences: list[str] = []
    for name, func in _public_methods(cls):
        for parameter in inspect.signature(func).parameters:
            if "user_id" in parameter:
                offences.append(f"{name}({parameter}=...)")
    assert not offences, (
        f"{cls.__module__}.{cls.__name__} exposes a per-call tenant override: "
        f"{', '.join(offences)}. The user id belongs in __init__ and nowhere else."
    )


@pytest.mark.parametrize("cls", SCOPED, ids=lambda c: f"{c.__module__}.{c.__name__}")
def test_scoped_repositories_require_a_user_at_construction(cls: type) -> None:
    parameters = inspect.signature(cls.__init__).parameters
    assert "user_id" in parameters, (
        f"{cls.__module__}.{cls.__name__} must take user_id in __init__ -- that is "
        "the only place it is allowed to appear."
    )
    assert parameters["user_id"].default is inspect.Parameter.empty, (
        f"{cls.__module__}.{cls.__name__} gives user_id a default in __init__, so "
        "an unscoped instance is constructible."
    )


@pytest.mark.parametrize("cls", SCOPED, ids=lambda c: f"{c.__module__}.{c.__name__}")
def test_scoped_repositories_offer_no_rebinding_escape_hatch(cls: type) -> None:
    names = {name for name, _ in _public_methods(cls)}
    assert not (names & _FORBIDDEN_METHOD_NAMES), (
        f"{cls.__module__}.{cls.__name__} exposes {sorted(names & _FORBIDDEN_METHOD_NAMES)}. "
        "A repository that can be re-pointed at another tenant is a repository "
        "someone can forget to re-point."
    )


@pytest.mark.parametrize(
    "cls",
    [c for c in ALL_REPOSITORIES if issubclass(c, PreAuthRepository)],
    ids=lambda c: f"{c.__module__}.{c.__name__}",
)
def test_pre_auth_repositories_justify_themselves(cls: type) -> None:
    """The exemption is narrow, so it has to be argued in the class's docstring."""
    doc = inspect.getdoc(cls) or ""
    assert len(doc) > 80, (
        f"{cls.__module__}.{cls.__name__} is exempt from tenancy scoping and must "
        "explain in its docstring why it cannot be scoped."
    )


def test_the_scheme_would_catch_a_violation() -> None:
    """The test above is only worth having if it can fail. Prove it does."""

    class LeakyRepository(TenantScopedRepository):
        def list_spans(self, user_id: str) -> list[str]:
            return []

    with pytest.raises(AssertionError, match="per-call tenant override"):
        test_no_scoped_method_accepts_a_user_id(LeakyRepository)
