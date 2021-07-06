from .naswot import NASWOT
from .proxy import Proxy

__all__ = ["get_proxy"]


def get_proxy(name: str) -> Proxy:
    if name == "NASWOT":
        return NASWOT()
