import pytest

from peft_reuse.backends import SyntheticBackend
from peft_reuse.demo import make_manifest


@pytest.fixture
def manifest():
    return make_manifest()


@pytest.fixture
def backend():
    return SyntheticBackend()
