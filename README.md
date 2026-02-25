# NJUxklogin

南京大学选课系统登录库，提供简洁的 Python API 完成选课系统的自动登录（含验证码识别）。

## 安装

```bash
pip install NJUxklogin
```

### 依赖

| 包名 | 用途 |
|------|------|
| `requests` | HTTP 请求 |
| `pillow` | 图像处理 |
| `numpy` | 矩阵计算 |
| `onnxruntime` | 验证码模型推理 |
| `pycryptodome` | DES 密码加密 |
| `pysocks` | SOCKS5 代理支持 |

## 快速开始

```python
from njuxklogin import login

# 基本用法：传入学号和密码，返回 (cookies字典, token字符串)
cookies, token = login("学号", "密码")

print(f"Token: {token}")
print(f"JSESSIONID: {cookies.get('JSESSIONID')}")
```

## API 文档

### `login(student_id, password, *, proxy=..., max_retries=None, force_refresh=False, use_cache=True)`

执行登录并返回凭证。

**参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `student_id` | `str` | ✅ | 学号（统一认证用户名） |
| `password` | `str` | ✅ | 密码明文（自动加密后提交） |
| `proxy` | `str \| None` | ❌ | 代理地址，如 `"socks5://127.0.0.1:1080"`。设置后会覆盖环境变量代理。不传则使用 `set_proxy()` 设置的全局值。 |
| `max_retries` | `int` | ❌ | 最大重试次数。不传则使用 `set_max_retries()` 设置的全局值（默认 8）。 |
| `force_refresh` | `bool` | ❌ | 是否忽略缓存强制重新登录（默认 `False`） |
| `use_cache` | `bool` | ❌ | 是否使用/更新登录态缓存（默认 `True`） |

**返回值：** `(cookies_dict, token)` 元组

**异常：**
- `ValueError`：参数缺失或无效
- `RuntimeError`：登录失败（达到最大重试次数）

### `set_proxy(proxy)`

设置全局代理地址，后续 `login()` 调用会默认使用该代理。

```python
from njuxklogin import set_proxy, login

set_proxy("socks5://127.0.0.1:1080")
cookies, token = login("学号", "密码")  # 自动使用上面设置的代理
```

传入 `None` 或空字符串可清除全局代理设置。

### `set_max_retries(max_retries)`

设置全局最大登录重试次数（默认 5）。

```python
from njuxklogin import set_max_retries, login

set_max_retries(10)
cookies, token = login("学号", "密码")  # 最多重试 10 次
```

### `get_cache(student_id)`

获取指定学号的缓存登录态。

```python
from njuxklogin import get_cache

cached = get_cache("学号")
if cached:
    cookies, token = cached
```

### `clear_cache(student_id=None)`

清除登录态缓存。传入学号则只清除该学号的缓存，不传则清除所有。

```python
from njuxklogin import clear_cache

clear_cache("学号")   # 清除指定学号
clear_cache()          # 清除所有
```

## 使用示例

### 基本登录

```python
from njuxklogin import login

try:
    cookies, token = login("20230001", "mypassword")
    print(f"登录成功！Token: {token}")
except RuntimeError as e:
    print(f"登录失败: {e}")
```

### 使用代理

```python
from njuxklogin import login

# 方式一：直接在 login 中传入
cookies, token = login("20230001", "mypassword", proxy="socks5://127.0.0.1:1080")

# 方式二：全局设置
from njuxklogin import set_proxy
set_proxy("http://127.0.0.1:7890")
cookies, token = login("20230001", "mypassword")
```

### 多账号登录

```python
from njuxklogin import login

# 缓存按学号自动区分，不会混淆
c1, t1 = login("20230001", "pwd1")
c2, t2 = login("20230002", "pwd2")

# 再次调用相同学号会直接返回缓存
c1_cached, t1_cached = login("20230001", "pwd1")  # 不会重新登录
assert t1 == t1_cached

# 强制重新登录
c1_new, t1_new = login("20230001", "pwd1", force_refresh=True)
```

## 返回值格式

`login()` 返回的格式与浏览器 Cookie/Token 一致：

```python
cookies = {
    "_WEU": "...",
    "JSESSIONID": "...",
    "route": "..."
}
token = "eyJ..."  # JWT Token
```

可直接用于 `requests` 库的后续请求：

```python
import requests

cookies, token = login("学号", "密码")
resp = requests.post(
    "https://xk.nju.edu.cn/xsxkapp/sys/xsxkapp/elective/volunteer.do",
    cookies=cookies,
    headers={"token": token},
    data={...},
    verify=False,
)
```

## 注意事项

- 需要在校内网络或通过 VPN/代理连接校园网
- 登录态缓存存储在内存中（全局变量），进程退出后失效
- 线程安全：多线程环境下可安全调用
- 验证码识别基于本地 ONNX 模型，无需外部 API

## 免责声明

本项目仅供学习交流使用。使用本工具产生的一切后果由使用者自行承担。
