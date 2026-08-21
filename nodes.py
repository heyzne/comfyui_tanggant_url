"""
ComfyUI 自定义节点：腾讯云 AIGC 生图

接口文档：腾讯云 AIGC 生图 — 外部对接文档 v2
- 提交任务：POST {base_url}/tencent-aigc-image
- 查询状态：GET {base_url}/tencent-aigc-image/status?taskId=xxx
- 认证：X-App-Id / X-Api-Key 请求头

v1.1.0 修复：
- 添加网络重试机制（指数退避），解决 SSL/TLS 握手不稳定导致的下载失败
- 使用 requests.Session + HTTPAdapter + urllib3.Retry 自动重试连接错误
- 为图片下载添加独立重试逻辑，适配 CDN 边缘节点的不稳定连接
- 增加更详细的错误日志，便于排查网络问题
"""

import io
import json
import os
import time
import warnings

import numpy as np
import requests
import torch
import urllib3
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

CATEGORY = "api/TencentAIGC"

DEFAULT_DEV_URL = "http://www.tongganagent.cn/api/v2/ai-creations"
DEFAULT_PROD_URL = "https://www.tongganai.com/api/v2/ai-creations"

# 配置文件（可选）：插件目录下的 config.json
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

# ---------------------------------------------------------------------------
# 网络重试配置
# ---------------------------------------------------------------------------

# 通用 API 请求重试策略（提交任务、查询状态）
API_RETRY_STRATEGY = Retry(
    total=5,                    # 总重试次数
    backoff_factor=1.0,         # 指数退避：1s, 2s, 4s, 8s...
    status_forcelist=[429, 500, 502, 503, 504],  # 这些 HTTP 状态码触发重试
    allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"],
    raise_on_status=False,
)

# 图片下载重试策略（CDN 边缘节点更不稳定，重试次数更多）
DOWNLOAD_RETRY_STRATEGY = Retry(
    total=8,                    # 更多重试次数
    backoff_factor=1.5,         # 稍慢的退避
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS"],
    raise_on_status=False,
)

# SSL 适配器：使用更兼容的 TLS 设置
class SSLAdapter(HTTPAdapter):
    """自定义适配器，允许更宽松的 SSL/TLS 握手，解决部分 CDN 的 EOF 问题"""
    def init_poolmanager(self, *args, **kwargs):
        import ssl
        context = ssl.create_default_context()
        # 允许 TLS 1.2/1.3，兼容旧服务器
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        # 某些 CDN 在握手阶段会异常断开，降低严格性
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED
        kwargs['ssl_context'] = context
        return super().init_poolmanager(*args, **kwargs)


def _create_session(retry_strategy: Retry, timeout: int = 30) -> requests.Session:
    """创建带重试机制的 Session"""
    session = requests.Session()
    adapter = SSLAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _parse_json(resp: requests.Response, action: str) -> dict:
    """解析 JSON 响应；非 JSON（如 413/502 的 HTML 错误页）时给出可读错误"""
    try:
        return resp.json()
    except ValueError:
        snippet = resp.text[:300].replace("\n", " ") if resp.text else "(空响应)"
        raise RuntimeError(
            f"{action}失败: HTTP {resp.status_code}，响应不是 JSON。"
            f"响应内容: {snippet}") from None


def _url_to_tensor(url: str, timeout: int = 120, max_retries: int = 8) -> torch.Tensor:
    """下载结果图 URL -> ComfyUI IMAGE tensor (1,H,W,C)

    修复：使用独立 Session + 指数退避重试，解决 CDN SSL 握手不稳定问题。
    """
    session = _create_session(DOWNLOAD_RETRY_STRATEGY, timeout=timeout)
    last_exception = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, timeout=timeout, stream=True)
            resp.raise_for_status()
            # 使用 stream=True 后手动读取内容，避免大图片内存问题
            content = resp.content
            pil = Image.open(io.BytesIO(content)).convert("RGB")
            arr = np.asarray(pil).astype(np.float32) / 255.0
            return torch.from_numpy(arr)[None, ...]
        except (requests.exceptions.SSLError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                urllib3.exceptions.SSLError,
                urllib3.exceptions.ProtocolError) as e:
            last_exception = e
            wait = min(2 ** attempt, 60)  # 指数退避，最大 60s
            print(f"[TencentAIGC] 下载图片失败 (尝试 {attempt}/{max_retries}): {type(e).__name__}: {e}")
            print(f"[TencentAIGC] 等待 {wait}s 后重试... URL: {url[:80]}...")
            time.sleep(wait)
        except Exception as e:
            # 非网络错误（如图片格式错误），直接抛出
            raise RuntimeError(f"下载图片失败（非网络错误）: {e}") from e

    raise RuntimeError(
        f"下载图片失败，已重试 {max_retries} 次。"
        f"最后错误: {type(last_exception).__name__}: {last_exception}\n"
        f"URL: {url}"
    ) from last_exception


def _blank_image(size: int = 512) -> torch.Tensor:
    return torch.zeros((1, size, size, 3), dtype=torch.float32)


def _load_file_config() -> dict:
    """从 config.json 读取默认凭证（可选）"""
    if os.path.isfile(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


# ---------------------------------------------------------------------------
# 节点 1：API 配置
# ---------------------------------------------------------------------------

class TencentAigcAPIConfig:
    """腾讯云 AIGC 接口凭证与调用参数配置"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "app_id": ("STRING", {"default": "", "multiline": False,
                                      "placeholder": "app_xxxxxxxx"}),
                "api_key": ("STRING", {"default": "", "multiline": False,
                                       "placeholder": "sk-xxxxxxxx"}),
                "environment": (["dev", "prod"], {"default": "dev"}),
                "base_url": ("STRING", {"default": "", "multiline": False,
                                        "placeholder": "留空则按 environment 使用默认地址"}),
                "model_name": ("STRING", {"default": "GG"}),
                "model_version": ("STRING", {"default": "3.1"}),
                "poll_interval": ("INT", {"default": 5, "min": 1, "max": 60,
                                           "tooltip": "状态轮询间隔（秒）"}),
                "max_wait": ("INT", {"default": 300, "min": 30, "max": 3600,
                                      "tooltip": "最长等待时间（秒）"}),
            }
        }

    RETURN_TYPES = ("AIGC_API_CONFIG",)
    RETURN_NAMES = ("api_config",)
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = "腾讯云 AIGC 接口配置（appId / apiKey / 环境 / 模型 / 轮询参数）"

    def build(self, app_id, api_key, environment, base_url,
              model_name, model_version, poll_interval, max_wait):
        if not base_url.strip():
            base_url = DEFAULT_DEV_URL if environment == "dev" else DEFAULT_PROD_URL
        config = {
            "app_id": app_id.strip(),
            "api_key": api_key.strip(),
            "base_url": base_url.strip().rstrip("/"),
            "model_name": model_name.strip() or "GG",
            "model_version": model_version.strip() or "3.1",
            "poll_interval": int(poll_interval),
            "max_wait": int(max_wait),
        }
        return (config,)


# ---------------------------------------------------------------------------
# 节点 2：AIGC 生图
# ---------------------------------------------------------------------------

class TencentAigcImage:
    """提交腾讯云 AIGC 生图任务并轮询等待结果，输出图片 / URL / 原始响应"""

    MAX_IMAGES = 14

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "api_config": ("AIGC_API_CONFIG",),
        }
        for i in range(1, cls.MAX_IMAGES + 1):
            optional[f"url{i}"] = ("STRING", {"forceInput": True,
                                              "tooltip": "参考图片的 URL 地址"})
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "resolution": (["1K", "2K", "4K"], {"default": "2K"}),
                "aspectRatio": (["empty", "1:1", "3:4", "4:3", "9:16", "16:9",
                                 "21:9", "4:5", "5:4", "3:2", "2:3"],
                                {"default": "empty",
                                 "tooltip": "empty = 传入空值，保持参考图原始比例"}),
                "skip_error": ("BOOLEAN", {"default": False,
                                            "tooltip": "开启后失败不中断，返回黑图，错误见 response"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                "control_after_generate": True,
                                "tooltip": "API 不支持种子；seed 用于派生 taskId，改 seed 即重新生成"}),
            },
            "optional": optional,
        }

    # 服务端当前支持的宽高比（非法/残留值视为 empty，不发送）
    VALID_RATIOS = {"1:1", "3:4", "4:3", "9:16", "16:9",
                    "21:9", "4:5", "5:4", "3:2", "2:3"}

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "url", "response")
    FUNCTION = "generate"
    CATEGORY = CATEGORY
    DESCRIPTION = "腾讯云 AIGC 生图：提交任务并轮询至完成，输出图片、URL 列表与原始响应"

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_config(api_config) -> dict:
        """优先使用 api_config 输入；未连接时回退到环境变量 / config.json"""
        if api_config and api_config.get("app_id") and api_config.get("api_key"):
            return api_config

        file_cfg = _load_file_config()
        return {
            "app_id": os.environ.get("TENCENT_AIGC_APP_ID", file_cfg.get("app_id", "")),
            "api_key": os.environ.get("TENCENT_AIGC_API_KEY", file_cfg.get("api_key", "")),
            "base_url": os.environ.get(
                "TENCENT_AIGC_BASE_URL",
                file_cfg.get("base_url", DEFAULT_DEV_URL)).rstrip("/"),
            "model_name": file_cfg.get("model_name", "GG"),
            "model_version": file_cfg.get("model_version", "3.1"),
            "poll_interval": int(file_cfg.get("poll_interval", 5)),
            "max_wait": int(file_cfg.get("max_wait", 300)),
        }

    def _submit(self, cfg: dict, task_id: int, prompt: str,
                resolution: str, aspect_ratio: str, input_files: list) -> dict:
        body = {
            "taskId": task_id,
            "prompt": prompt,
            "resolution": resolution,
            "inputFiles": input_files,
            "modelName": cfg["model_name"],
            "modelVersion": cfg["model_version"],
        }
        ratio = (aspect_ratio or "").strip()
        if ratio in self.VALID_RATIOS:
            body["aspectRatio"] = ratio
        else:
            # empty / 旧工作流残留值：传空字符串，让服务端保持参考图原始比例
            if ratio not in ("", "empty"):
                print(f"[TencentAIGC] 忽略无效的 aspectRatio={ratio!r}，"
                      f"改为传入空值（保持原始比例）")
            body["aspectRatio"] = ""

        session = _create_session(API_RETRY_STRATEGY)
        try:
            resp = session.post(
                f"{cfg['base_url']}/tencent-aigc-image",
                json=body,
                headers={
                    "Content-Type": "application/json",
                    "X-App-Id": cfg["app_id"],
                    "X-Api-Key": cfg["api_key"],
                },
                timeout=30,
            )
            return _parse_json(resp, "提交任务")
        except requests.exceptions.RetryError as e:
            raise RuntimeError(f"提交任务失败，已重试多次: {e}") from e

    def _poll(self, cfg: dict, tencent_task_id: str) -> dict:
        """轮询直到 FINISH / FAIL / ABORTED / 超时，返回最后一次状态响应"""
        url = f"{cfg['base_url']}/tencent-aigc-image/status"
        headers = {"X-App-Id": cfg["app_id"], "X-Api-Key": cfg["api_key"]}
        deadline = time.time() + cfg["max_wait"]
        last = {}
        last_status = None

        session = _create_session(API_RETRY_STRATEGY)

        while time.time() < deadline:
            time.sleep(cfg["poll_interval"])
            try:
                resp = session.get(url, params={"taskId": tencent_task_id},
                                   headers=headers, timeout=30)
                last = _parse_json(resp, "查询状态")
            except requests.exceptions.RetryError as e:
                print(f"[TencentAIGC] 查询状态重试耗尽: {e}")
                continue

            if last.get("code") != 200:
                raise RuntimeError(f"查询状态失败: {last.get('message')}")
            status = last.get("data", {}).get("status", "")
            if status != last_status:  # 只在状态变化时打印一次
                print(f"[TencentAIGC] 任务 {tencent_task_id} 状态: {status}")
                last_status = status
            if status in ("FINISH", "FAIL", "ABORTED"):
                return last

        raise TimeoutError(f"轮询超时（{cfg['max_wait']}s），任务仍在处理中: {tencent_task_id}")

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def generate(self, prompt, resolution, aspectRatio, skip_error, seed,
                 api_config=None, **kwargs):
        try:
            return self._run(prompt, resolution, aspectRatio, seed, api_config, kwargs)
        except Exception as e:
            if not skip_error:
                raise
            err = {"error": str(e)}
            print(f"[TencentAIGC] 生图失败（skip_error 已开启）: {e}")
            return (_blank_image(), "", json.dumps(err, ensure_ascii=False, indent=2))

    def _run(self, prompt, resolution, aspect_ratio, seed, api_config, kwargs):
        cfg = self._resolve_config(api_config)
        if not cfg["app_id"] or not cfg["api_key"]:
            raise ValueError(
                "缺少 appId / apiKey：请连接 api_config 节点，"
                "或设置环境变量 TENCENT_AIGC_APP_ID / TENCENT_AIGC_API_KEY，"
                "或在插件目录创建 config.json")

        # 收集参考图 URL -> inputFiles（跳过未连接 / 空白输入）
        input_files = []
        for i in range(1, self.MAX_IMAGES + 1):
            url = kwargs.get(f"url{i}")
            if url and url.strip():
                input_files.append({"url": url.strip()})

        # taskId：seed + 时间戳派生，保证同一 seed 可复现、不同 seed 触发重跑
        task_id = int((seed + int(time.time() * 1000)) % 2_000_000_000)

        # 1. 提交任务
        submit_resp = self._submit(cfg, task_id, prompt, resolution,
                                   aspect_ratio, input_files)
        if submit_resp.get("code") != 200:
            raise RuntimeError(f"提交任务失败: {submit_resp.get('message')}")

        tencent_task_id = submit_resp["data"]["tencentTaskId"]
        print(f"[TencentAIGC] 任务已提交 taskId={task_id} "
              f"tencentTaskId={tencent_task_id} 参考图 {len(input_files)} 张")

        # 2. 轮询等待
        final = self._poll(cfg, tencent_task_id)
        data = final.get("data", {})

        if data.get("status") != "FINISH":
            raise RuntimeError(
                f"任务未成功: status={data.get('status')} "
                f"message={data.get('message')} "
                f"errCode={data.get('errCode')} errCodeExt={data.get('errCodeExt')}")

        image_urls = data.get("imageUrls") or []
        if not image_urls:
            raise RuntimeError("任务已完成但未返回图片 URL")

        # 3. 下载结果图（带重试）
        print(f"[TencentAIGC] 开始下载 {len(image_urls)} 张结果图...")
        frames = []
        for idx, u in enumerate(image_urls, 1):
            print(f"[TencentAIGC] 下载第 {idx}/{len(image_urls)} 张...")
            frames.append(_url_to_tensor(u))
        image_batch = torch.cat(frames, dim=0)
        print(f"[TencentAIGC] 全部 {len(image_urls)} 张图片下载完成")

        response = json.dumps(
            {"submit": submit_resp, "final": final},
            ensure_ascii=False, indent=2)
        return (image_batch, "\n".join(image_urls), response)


# ---------------------------------------------------------------------------
# 注册
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "TencentAigcAPIConfig": TencentAigcAPIConfig,
    "TencentAigcImage": TencentAigcImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TencentAigcAPIConfig": "Tencent AIGC Config",
    "TencentAigcImage": "Tencent AIGC Image",
}
