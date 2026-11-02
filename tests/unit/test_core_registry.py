import pytest

from core.errors import ValidationError
from core.registry import Registry


def test_register_and_get():
    reg = Registry("thing")

    @reg.register("foo")
    class Foo:
        pass

    assert reg.get("foo") is Foo
    assert reg.list() == ["foo"]


def test_duplicate_registration_raises():
    reg = Registry("thing")

    @reg.register("foo")
    class Foo:
        pass

    with pytest.raises(ValidationError):
        @reg.register("foo")
        class Bar:
            pass


def test_unknown_lookup_raises_with_helpful_message():
    reg = Registry("thing")
    with pytest.raises(ValidationError, match="Unknown thing"):
        reg.get("missing")
