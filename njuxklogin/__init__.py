"""
NJUxklogin - 南京大学选课系统登录库

使用方法:
    from njuxklogin import login, set_proxy, set_max_retries

    # 基本用法
    cookies, token = login("学号", "密码")

    # 设置代理（可选，会覆盖环境变量代理）
    set_proxy("socks5://127.0.0.1:1080")
    cookies, token = login("学号", "密码")

    # 设置最大重试次数（可选，默认8）
    set_max_retries(10)

    # 也可以在 login 时直接传入
    cookies, token = login("学号", "密码", proxy="socks5://127.0.0.1:1080", max_retries=10)
"""

import threading
from typing import Dict, Optional, Tuple

from njuxklogin._authenticator import perform_login as _perform_login

__version__ = "1.0.0"

# ================= 全局配置 =================
_config_lock = threading.Lock()
_global_proxy: Optional[str] = None
_global_max_retries: int = 8

# ================= 登录态缓存（按学号区分） =================
_cache_lock = threading.Lock()
_session_cache: Dict[str, Tuple[dict, str]] = {}
# 格式: { "学号": (cookies_dict, token) }


def set_proxy(proxy: Optional[str]) -> None:
    """设置全局代理地址。

    设置后，后续所有 login() 调用都会使用该代理（除非 login() 中显式传入 proxy 覆盖）。
    传入 None 或空字符串可清除代理设置。

    如果设置了代理，会覆盖环境变量中的代理配置。

    Args:
        proxy: 代理地址，如 "socks5://127.0.0.1:1080" 或 "http://127.0.0.1:7890"
    """
    global _global_proxy
    with _config_lock:
        _global_proxy = proxy.strip() if proxy and proxy.strip() else None


def set_max_retries(max_retries: int) -> None:
    """设置全局最大登录重试次数。

    设置后，后续所有 login() 调用都会使用该值（除非 login() 中显式传入 max_retries 覆盖）。

    Args:
        max_retries: 最大重试次数，必须 >= 1
    """
    global _global_max_retries
    if not isinstance(max_retries, int) or max_retries < 1:
        raise ValueError(f"max_retries 必须是 >= 1 的整数，实际: {max_retries}")
    with _config_lock:
        _global_max_retries = max_retries


def get_cache(student_id: str) -> Optional[Tuple[dict, str]]:
    """获取指定学号的缓存登录态。

    Returns:
        (cookies_dict, token) 或 None（无缓存时）
    """
    with _cache_lock:
        return _session_cache.get(student_id)


def clear_cache(student_id: Optional[str] = None) -> None:
    """清除登录态缓存。

    Args:
        student_id: 指定学号则只清除该学号的缓存；None 则清除所有缓存。
    """
    with _cache_lock:
        if student_id is None:
            _session_cache.clear()
        else:
            _session_cache.pop(student_id, None)


def login(
    student_id: str,
    password: str,
    *,
    proxy: Optional[str] = ...,
    max_retries: Optional[int] = None,
    force_refresh: bool = False,
    use_cache: bool = True,
) -> Tuple[dict, str]:
    """登录南京大学选课系统。

    Args:
        student_id: 学号（统一认证用户名）
        password: 密码明文（会自动加密后提交）
        proxy: 代理地址（可选）。传入后会覆盖环境变量代理。
               不传则使用 set_proxy() 设置的全局代理。
               传入 None 表示不使用代理。
        max_retries: 最大重试次数（可选，默认使用 set_max_retries() 设置的值，初始为 5）
        force_refresh: 是否强制重新登录（忽略缓存）
        use_cache: 是否使用/更新缓存（默认 True）

    Returns:
        (cookies_dict, token) 元组

    Raises:
        ValueError: 参数缺失或无效
        RuntimeError: 登录失败（达到最大重试次数）
        ConnectionError: 网络连接失败
    """
    # 参数校验
    if not student_id or not isinstance(student_id, str):
        raise ValueError("student_id 不能为空")
    if not password or not isinstance(password, str):
        raise ValueError("password 不能为空")

    student_id = student_id.strip()
    password = password.strip()

    # 确定代理和重试次数
    with _config_lock:
        effective_proxy = _global_proxy if proxy is ... else (
            proxy.strip() if proxy and proxy.strip() else None
        )
        effective_retries = max_retries if max_retries is not None else _global_max_retries

    if not isinstance(effective_retries, int) or effective_retries < 1:
        raise ValueError(f"max_retries 必须是 >= 1 的整数，实际: {effective_retries}")

    # 检查缓存
    if use_cache and not force_refresh:
        cached = get_cache(student_id)
        if cached is not None:
            return cached

    # 执行登录
    cookies, token = _perform_login(
        username=student_id,
        password=password,
        proxy=effective_proxy,
        max_retries=effective_retries,
    )

    if not cookies or not token:
        raise RuntimeError(
            f"登录失败：已重试 {effective_retries} 次仍未成功。"
            f"请检查学号密码是否正确、网络是否可达。"
        )

    # 更新缓存
    if use_cache:
        with _cache_lock:
            _session_cache[student_id] = (cookies, token)

    return cookies, token
