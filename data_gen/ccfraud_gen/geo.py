from __future__ import annotations

import ipaddress # standard library for working with IPv4/IPv6 networks/addresses
import random # deterministic sampling when you pass a seeded random.Random
from dataclasses import dataclass # easy way to define simple “record” objects

# A simple representation of a country with some geographic and networking info.
@dataclass(frozen=True)
class Country:
    code: str
    name: str
    # A representative bounding box (lat_min, lat_max, lon_min, lon_max)
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    # A private-ish CIDR used only for synthetic "GeoIP-like" mapping in this generator
    cidr: str


# A compact, opinionated set of countries to keep things manageable.
# Expand later if you want more diversity.
COUNTRIES: list[Country] = [
    Country("US", "United States", 25.0, 49.0, -124.0, -66.0, "10.0.0.0/8"),
    Country("CA", "Canada", 42.0, 70.0, -140.0, -52.0, "172.16.0.0/12"),
    Country("MX", "Mexico", 14.0, 33.0, -118.0, -86.0, "192.168.0.0/16"),
    Country("GB", "United Kingdom", 49.9, 58.7, -8.6, 1.8, "100.64.0.0/10"),
    Country("FR", "France", 41.0, 51.5, -5.5, 9.6, "198.18.0.0/15"),
    Country("DE", "Germany", 47.2, 55.1, 5.9, 15.0, "198.51.100.0/24"),
    Country("ES", "Spain", 36.0, 43.8, -9.3, 3.3, "203.0.113.0/24"),
    Country("IT", "Italy", 36.6, 47.1, 6.6, 18.5, "169.254.0.0/16"),
    Country("SE", "Sweden", 55.0, 69.0, 11.0, 24.0, "240.0.0.0/8"),
]

COUNTRY_BY_CODE = {c.code: c for c in COUNTRIES}

# A simple neighbor graph for "regional travel" probabilities.
NEIGHBORS: dict[str, list[str]] = {
    "US": ["CA", "MX"],
    "CA": ["US"],
    "MX": ["US"],
    "GB": ["FR"],
    "FR": ["GB", "DE", "ES", "IT"],
    "DE": ["FR"],
    "ES": ["FR"],
    "IT": ["FR"],
    "SE": ["DE"],
}

# Pick a random Country from COUNTRIES, returns its .code
def random_country_code(rng: random.Random) -> str:
    return rng.choice(COUNTRIES).code


def pick_next_country(
    rng: random.Random,
    home: str,
    current: str,
    p_home: float,
    p_neighbor: float,
    p_international: float,
) -> str:
    # Normalize (in case caller changes params)
    s = p_home + p_neighbor + p_international
    p_home, p_neighbor, p_international = p_home / s, p_neighbor / s, p_international / s

    r = rng.random()
    if r < p_home:
        return home
    if r < p_home + p_neighbor:
        # pick from neighbors of current or home
        candidates = NEIGHBORS.get(current) or NEIGHBORS.get(home) or [home]
        return rng.choice(candidates)
    # international: any country except current
    codes = [c.code for c in COUNTRIES if c.code != current]
    return rng.choice(codes)


def sample_lat_lon(rng: random.Random, country_code: str) -> tuple[float, float]:
    """
    Samples a random point inside the bounding box of the given country.
    Important caveat: bounding boxes include points outside the true country border 
    (e.g., ocean/neighboring areas). OK for synthetic data.
    """
    c = COUNTRY_BY_CODE[country_code]
    lat = rng.uniform(c.lat_min, c.lat_max)
    lon = rng.uniform(c.lon_min, c.lon_max)
    return round(lat, 6), round(lon, 6)


def sample_ip(rng: random.Random, country_code: str) -> str:
    """Sample an IP within a synthetic CIDR assigned to the country.
    Generates a random IPv4 address from the country’s assigned CIDR."""
    c = COUNTRY_BY_CODE[country_code]
    net = ipaddress.ip_network(c.cidr, strict=False)
    # avoid network/broadcast where applicable
    # For very small nets (e.g. /24), this still works.
    first = int(net.network_address) + 1
    last = int(net.broadcast_address) - 1
    if last <= first:
        ip_int = first
    else:
        ip_int = rng.randint(first, last)
    return str(ipaddress.ip_address(ip_int))
