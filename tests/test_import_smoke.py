from __future__ import annotations

import bittrace
import bittrace.experimental
import bittrace.source
import bittrace.v3


def test_public_imports_resolve() -> None:
    assert bittrace.__version__ == "0.3.1"
    assert bittrace.__file__
    assert bittrace.source.__file__
    assert bittrace.experimental.__file__
    assert bittrace.v3.__file__
    assert hasattr(bittrace.v3, "ContractValidationError")
    assert hasattr(bittrace.source, "prepare_full_binary_campaign")
