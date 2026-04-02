"""URL validation to prevent SSRF attacks."""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse


class SSRFError(ValueError):
    """Raised when a URL targets an internal/private network."""


# Private/internal IP ranges that should never be accessed
_BLOCKED_NETWORKS = [
    # IPv4
    ipaddress.ip_network("127.0.0.0/8"),  # Loopback
    ipaddress.ip_network("10.0.0.0/8"),  # Private Class A
    ipaddress.ip_network("172.16.0.0/12"),  # Private Class B
    ipaddress.ip_network("192.168.0.0/16"),  # Private Class C
    ipaddress.ip_network("169.254.0.0/16"),  # Link-local (AWS metadata)
    ipaddress.ip_network("0.0.0.0/8"),  # "This" network
    # IPv6
    ipaddress.ip_network("::1/128"),  # IPv6 loopback
    ipaddress.ip_network("fe80::/10"),  # IPv6 link-local
    ipaddress.ip_network("fc00::/7"),  # IPv6 unique local
    ipaddress.ip_network("::ffff:0:0/96"),  # IPv4-mapped IPv6
]

_BLOCKED_HOSTNAMES = {"localhost", "0.0.0.0", "::1"}

_ALLOWED_SCHEMES = {"http", "https"}


def validate_url(url: str) -> None:
    """Validate that a URL doesn't target internal/private networks.

    Raises SSRFError if the URL is internal, private, or malformed.
    """
    if not url:
        raise SSRFError("Empty URL")

    parsed = urlparse(url)

    # Check scheme
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise SSRFError(f"Scheme '{parsed.scheme}' not allowed, must be http or https")

    hostname = parsed.hostname
    if not hostname:
        raise SSRFError("No hostname in URL")

    # Check blocked hostnames
    if hostname.lower() in _BLOCKED_HOSTNAMES:
        raise SSRFError(f"URL targets internal host: {hostname}")

    # Check if hostname is an IP address in a blocked range
    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        # Not an IP address — it's a domain name, allow it
        # (DNS resolution to private IPs is a separate concern for production hardening)
        return

    for network in _BLOCKED_NETWORKS:
        if ip in network:
            raise SSRFError(f"URL targets internal network: {hostname}")
