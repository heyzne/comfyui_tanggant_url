# comfyui_tanggant_url

腾讯云 AIGC 生图 ComfyUI 自定义节点（URL 参考图版），基于《腾讯云 AIGC 生图 — 外部对接文档 v2》实现：提交生图任务 → 轮询状态 → 下载结果图。

## 节点

### 1. Tencent AIGC Config（`api/TencentAIGC`）

输出 `api_config`，配置凭证与调用参数：

| 参数 | 说明 |
| --- | --- |
| `app_id` / `api_key` | 管理员分配的凭证（`app_xxxxxxxx` / `sk-...`） |
| `environment` | `dev`（tongganagent.cn）/ `prod`（tongganai.com） |
| `base_url` | 留空则按 environment 使用默认地址，也可自定义网关地址 |
| `model_name` / `model_version` | 默认 `GG` / `3.1` |
| `poll_interval` | 状态轮询间隔（秒），默认 5 |
| `max_wait` | 最长等待时间（秒），默认 300 |

### 2. Tencent AIGC Image（`api/TencentAIGC`）

主生图节点：

- **输入**
  - `url1` ~ `url14`（可选）：参考图片的 URL 地址（公网可访问的短 URL，接口限制最长 8182 字符），逐个放入 `inputFiles`
  - `api_config`（可选）：来自 Config 节点；未连接时回退到环境变量或 `config.json`
- **控件**
  - `prompt`：生图提示词
  - `resolution`：`1K` / `2K` / `4K`
  - `aspectRatio`：`empty`（不发送，用服务端默认）/ `1:1` / `3:4` / `4:3` / `9:16` / `16:9`
  - `skip_error`：开启后失败不中断工作流，返回黑图，错误详情见 `response`
  - `seed` + `control_after_generate`：API 本身不支持种子，seed 用于派生 `taskId`；改 seed 即触发重新生成
- **输出**
  - `image`：生成的图片（多张时自动合并为 batch）
  - `url`：图片 URL 列表（每行一个）
  - `response`：提交与最终状态的原始 JSON（调试用）

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/heyzen/comfyui_tanggant_url.git
pip install -r comfyui_tanggant_url/requirements.txt
```

`requests`、`Pillow`、`numpy`、`torch` 在 ComfyUI 环境中一般已自带。重启 ComfyUI 后，在节点菜单 `api/TencentAIGC` 下即可找到。

## 凭证的三种提供方式（优先级从高到低）

1. 连接 **Tencent AIGC Config** 节点的 `api_config` 输出；
2. 环境变量：`TENCENT_AIGC_APP_ID`、`TENCENT_AIGC_API_KEY`、`TENCENT_AIGC_BASE_URL`；
3. 插件目录下的 `config.json`（参考 `config.example.json` 复制改名填写）。

## 说明

- 任务通常 30 秒 ~ 2 分钟完成，节点按 `poll_interval` 轮询直至 `FINISH` / `FAIL` / 超时。
- 接口限流默认 60 次/分钟/appId，超出返回 429，请稍后重试。
- `aspectRatio` 选 `empty` 时请求体不含该字段，由服务端决定默认比例。
- 参考图通过 `url1` ~ `url14` 直接传入图片 URL（需公网可访问，接口限制 url 最长 8182 字符，不支持 base64 data URI），原样放入 `inputFiles` 提交。
- 接口返回非 JSON（如 413/502 的 HTML 错误页）时，节点会报出 HTTP 状态码和响应片段，便于排查。

## License

MIT © [heyzen](https://github.com/heyzen)
