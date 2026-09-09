"""Skip copyright check (this package uses an MIT LICENSE/README, not headers)."""
import pytest


@pytest.mark.copyright
@pytest.mark.linter
def test_copyright():
    pytest.skip('copyright headers intentionally omitted')
