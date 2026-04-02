"""Tests for SSRF protection on URL validation."""

import pytest

from agentic.agent.url_validation import SSRFError, validate_url


class TestValidateUrl:
    def test_allows_public_url(self):
        validate_url("https://api.example.com/endpoint")  # Should not raise

    def test_allows_public_ip(self):
        validate_url("https://203.0.113.1/api")  # Public IP, should not raise

    def test_blocks_localhost(self):
        with pytest.raises(SSRFError, match="internal"):
            validate_url("http://localhost/admin")

    def test_blocks_localhost_127(self):
        with pytest.raises(SSRFError, match="internal"):
            validate_url("http://127.0.0.1/admin")

    def test_blocks_aws_metadata(self):
        with pytest.raises(SSRFError, match="internal"):
            validate_url("http://169.254.169.254/latest/meta-data/")

    def test_blocks_link_local(self):
        with pytest.raises(SSRFError, match="internal"):
            validate_url("http://169.254.1.1/something")

    def test_blocks_private_10(self):
        with pytest.raises(SSRFError, match="internal"):
            validate_url("http://10.0.0.1/internal-api")

    def test_blocks_private_172(self):
        with pytest.raises(SSRFError, match="internal"):
            validate_url("http://172.16.0.1/internal")

    def test_blocks_private_192(self):
        with pytest.raises(SSRFError, match="internal"):
            validate_url("http://192.168.1.1/router")

    def test_blocks_zero_ip(self):
        with pytest.raises(SSRFError, match="internal"):
            validate_url("http://0.0.0.0/")

    def test_allows_http_and_https(self):
        validate_url("http://example.com/api")
        validate_url("https://example.com/api")

    def test_blocks_non_http_schemes(self):
        with pytest.raises(SSRFError):
            validate_url("file:///etc/passwd")

    def test_blocks_empty_url(self):
        with pytest.raises(SSRFError):
            validate_url("")

    def test_blocks_no_scheme(self):
        with pytest.raises(SSRFError):
            validate_url("example.com/api")
