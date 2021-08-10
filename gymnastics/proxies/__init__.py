from .naswot import NASWOT
from .fisher import Fisher
from .proxy import Proxy

__all__ = ["get_proxy"]


def get_proxy(name: str) -> Proxy:
    if name == "NASWOT":
        return NASWOT()
    elif name == "Fisher":
        return Fisher()
