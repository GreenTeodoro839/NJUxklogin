"""
NJUxklogin 内部登录流程实现。
不依赖配置文件，所有参数通过函数参数传入。
"""

import os
import time
from typing import Optional, Tuple

import requests

from njuxklogin._des_encrypt import encrypt_password
from njuxklogin._captcha import solve_captcha_from_base64

BASE_URL = "https://xk.nju.edu.cn/xsxkapp/sys/xsxkapp"
INDEX_URL = f"{BASE_URL}/*default/index.do"
VCODE_API = f"{BASE_URL}/student/4/vcode.do"
LOGIN_API = f"{BASE_URL}/student/check/login.do"


def _build_proxies(proxy: Optional[str]):
    if not proxy:
        return None
    if proxy.startswith("socks5://"):
        proxy = proxy.replace("socks5://", "socks5h://", 1)
    return {"http": proxy, "https": proxy}


def perform_login(
    *,
    username: str,
    password: str,
    proxy: Optional[str] = None,
    max_retries: int = 8,
) -> Tuple[Optional[dict], Optional[str]]:
    """执行完整登录流程。

    Args:
        username: 学号
        password: 密码明文
        proxy: 代理地址（可选）
        max_retries: 最大重试次数

    Returns:
        (cookies_dict, token) 或 (None, None)
    """
    encrypted_pwd = encrypt_password(password)
    proxies = _build_proxies(proxy)

    def _new_session():
        s = requests.Session()
        s.trust_env = False  # 不使用环境变量代理
        s.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36",
            "Referer": INDEX_URL,
            "Origin": "https://xk.nju.edu.cn",
            "X-Requested-With": "XMLHttpRequest",
        })
        if proxies:
            s.proxies = proxies
        return s

    # 如果设置了代理，覆盖环境变量代理
    if proxy:
        for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
                     "ALL_PROXY", "all_proxy"):
            os.environ.pop(var, None)

    session = _new_session()

    for attempt in range(max_retries):
        try:
            print(f"\n====== 尝试第 {attempt + 1}/{max_retries} 次登录 ======")

            # Step 1: 初始化 Session
            print(">>> 1. 初始化 Session...")
            session.get(INDEX_URL, timeout=10)

            # Step 2: 获取验证码
            print(">>> 2. 获取验证码...")
            vcode_resp = session.post(VCODE_API, timeout=10)
            vcode_json = vcode_resp.json()

            data_node = vcode_json.get("data", {})
            server_uuid = data_node.get("uuid")
            img_b64_raw = data_node.get("vode") or data_node.get("vcode")

            if not server_uuid or not img_b64_raw:
                print(f"❌ 响应数据不完整: {vcode_json}")
                continue

            img_gif_b64_body = img_b64_raw.split(",")[1] if "," in img_b64_raw else img_b64_raw

            # Step 3: 识别验证码
            print(">>> 3. 识别验证码...")
            points = solve_captcha_from_base64(img_gif_b64_body)
            if not points:
                print("❌ 识别失败")
                continue

            coord_str_list = [f"{int(p[0])}-{int(p[1] * 5 / 6)}" for p in points]
            verify_code = ",".join(coord_str_list)
            print(f"    提交坐标: {verify_code}")

            # Step 4: 发送登录请求
            payload = {
                "loginName": username,
                "loginPwd": encrypted_pwd,
                "verifyCode": verify_code,
                "vtoken": "",
                "uuid": server_uuid,
            }

            print(">>> 4. 发送登录请求...")
            login_resp = session.post(LOGIN_API, data=payload, timeout=15)
            login_json = login_resp.json()

            # Step 5: 结果校验
            resp_code = login_json.get("code")
            resp_data = login_json.get("data") or {}

            if str(resp_code) == "1" and str(resp_data.get("number")) == str(username):
                token = resp_data.get("token")
                print(f"✅ 登录成功!")
                return session.cookies.get_dict(), token
            else:
                msg = login_json.get("msg", "未知错误")
                print(f"❌ 登录失败: {msg} (Code: {resp_code})")

                # 服务端拒绝当前会话时重建 Session
                if str(resp_code).startswith("#E"):
                    print("⚠️  服务端拒绝当前会话，正在重建 Session...")
                    session.close()
                    session = _new_session()

        except requests.ConnectionError as e:
            print(f"❌ 网络连接失败: {e}")
            time.sleep(1)
        except requests.Timeout:
            print("❌ 请求超时")
            time.sleep(1)
        except Exception as e:
            print(f"❌ 异常: {e}")
            time.sleep(1)

    print("🚫 登录失败，已达最大重试次数。")
    return None, None
