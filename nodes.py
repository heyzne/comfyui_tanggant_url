"""
ComfyUI 自定义节点：腾讯云 AIGC 生图

接口文档：腾讯云 AIGC 生图 — 外部对接文档 v2
- 提交任务：POST {base_url}/tencent-aigc-image
- 查询状态：GET  {base_url}/tencent-aigc-image/status?taskId=xxx
- 认证：X-App-Id / X-Api-Key 请求头
"""

import io
import json
import os
import time

import numpy as np
import requests
import torch
from PIL import Image

CATEGORY = "api/TencentAIGC"

DEFAULT_DEV_URL = "http://www.tongganagent.cn/api/v2/ai-creations"
DEFAULT_PROD_URL = "https://www.tongganai.com/api/v2/ai-creations"

# 配置文件（可选）：插件目录下的 config.json
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


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


def _url_to_tensor(url: str, timeout: int = 120) -> torch.Tensor:
    """下载结果图 URL -> ComfyUI IMAGE tensor (1,H,W,C)"""
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    pil = Image.open(io.BytesIO(resp.content)).convert("RGB")
    arr = np.asarray(pil).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]


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
                "aspectRatio": (["empty", "1:1", "3:4", "4:3", "9:16", "16:9"],
                                {"default": "empty",
                                 "tooltip": "empty = 不发送该字段，使用服务端默认"}),
                "skip_error": ("BOOLEAN", {"default": False,
                                           "tooltip": "开启后失败不中断，返回黑图，错误见 response"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "control_after_generate": True,
                                 "tooltip": "API 不支持种子；seed 用于派生 taskId，改 seed 即重新生成"}),
            },
            "optional": optional,
        }

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
        if aspect_ratio != "empty":
            body["aspectRatio"] = aspect_ratio

        resp = requests.post(
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

    def _poll(self, cfg: dict, tencent_task_id: str) -> dict:
        """轮询直到 FINISH / FAIL / ABORTED / 超时，返回最后一次状态响应"""
        url = f"{cfg['base_url']}/tencent-aigc-image/status"
        headers = {"X-App-Id": cfg["app_id"], "X-Api-Key": cfg["api_key"]}
        deadline = time.time() + cfg["max_wait"]
        last = {}

        while time.time() < deadline:
            time.sleep(cfg["poll_interval"])
            resp = requests.get(url, params={"taskId": tencent_task_id},
                                headers=headers, timeout=30)
            last = _parse_json(resp, "查询状态")
            if last.get("code") != 200:
                raise RuntimeError(f"查询状态失败: {last.get('message')}")
            status = last.get("data", {}).get("status", "")
            print(f"[TencentAIGC] 任务 {tencent_task_id} 状态: {status}")
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

        # 3. 下载结果图
        frames = [_url_to_tensor(u) for u in image_urls]
        image_batch = torch.cat(frames, dim=0)

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
