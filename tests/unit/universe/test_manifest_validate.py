from src.universe.manifest_validate import validate_deployment_manifest


def test_validate_deployment_manifest_null_is_provisional() -> None:
    result = validate_deployment_manifest(manifest=None, present_tickers=frozenset({"069500"}), issuer_by_ticker={"069500": "삼성자산운용"}, allowed_issuers=frozenset({"삼성자산운용"}))
    assert result.status == "PROVISIONAL"


def test_validate_deployment_manifest_missing_and_issuer_fail_closed() -> None:
    import pytest

    from src.universe.manifest_validate import ManifestValidationError

    with pytest.raises(ManifestValidationError):
        validate_deployment_manifest(manifest=frozenset(), present_tickers=frozenset({"069500"}), issuer_by_ticker={"069500": "삼성자산운용"}, allowed_issuers=frozenset({"삼성자산운용"}))
    with pytest.raises(ManifestValidationError):
        validate_deployment_manifest(manifest=frozenset({"233740"}), present_tickers=frozenset({"069500"}), issuer_by_ticker={"069500": "삼성자산운용"}, allowed_issuers=frozenset({"삼성자산운용"}))
    with pytest.raises(ManifestValidationError):
        validate_deployment_manifest(manifest=frozenset({"069500"}), present_tickers=frozenset({"069500"}), issuer_by_ticker={"069500": "UNKNOWN"}, allowed_issuers=frozenset({"삼성자산운용"}))
    ok = validate_deployment_manifest(manifest=frozenset({"069500"}), present_tickers=frozenset({"069500"}), issuer_by_ticker={"069500": "삼성자산운용"}, allowed_issuers=frozenset({"삼성자산운용"}))
    assert ok.status == "ENFORCED"
