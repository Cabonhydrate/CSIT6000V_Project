# TACC 最短可运行流程（公共版 · CSIT6000V）

> 适合已经完成 `tcloud` 账号配置的用户。首次使用、Windows WSL2 配置、Private vLLM 以及报错处理，请阅读 [README_TACC.md](README_TACC.md)。**本流程从未在你的 TACC 账号上真实提交**，请以实际返回的 Job ID、节点地址为准。

## 初次准备（WSL2 Bash）

1. 安装课程 `tcloud` v0.4.1 SDK（Linux x86_64），确保 `tcloud --help` 能执行。
2. 将 `project-release-1(1).zip` 解压到一个独立目录，在含 `setup-sdk-account.sh` 的根目录中执行：

```bash
tcloud config -u YOUR_TACC_USERNAME -f "$HOME/.ssh/YOUR_PRIVATE_KEY"
tcloud init
tcloud env ls
PROJECT_DIR=$(pwd -P)
bash setup-sdk-account.sh
```

**不要**与私有版（`mpi-gemm-v1`）混用。公共版的管理环境是 `proxy`。

## 每次运行（WSL2 Bash）

```bash
# 1. 先定位目录；新开终端时重新设置 PROJECT_DIR
cd ~/csit6000v-tacc/qwen27b-student-release
PROJECT_DIR=$(pwd -P)

# 2. 提交 GPU 模型服务
cd "$PROJECT_DIR/qwen-public-serve"
tcloud submit

# 3. 把刚才输出的作业 ID 填在这里
SERVICE_JOB_ID=填写GPU服务作业ID
tcloud ps -j "$SERVICE_JOB_ID"
tcloud cat "slurm_log/qwen-public-serve-${SERVICE_JOB_ID}.log"

# 4. 必须看到 Application startup complete
#    复制日志 SERVICE_BASE_URL= 后面的实际地址
cd "$PROJECT_DIR"
SERVICE_BASE_URL='http://本次实际计算节点:8000/v1'
printf '%s\n' "$SERVICE_BASE_URL" > qwen-public-client/server-url.txt

# 5. 提交另一个 CPU Client Job，调用 GPU 服务
cd "$PROJECT_DIR/qwen-public-client"
tcloud submit
CLIENT_JOB_ID=填写CPU客户端作业ID
tcloud ps -j "$CLIENT_JOB_ID"
tcloud cat "slurm_log/qwen-public-client-${CLIENT_JOB_ID}.log"

# 6. 下载响应 JSON、日志
cd "$PROJECT_DIR"
tcloud download "slurm_log/qwen-public-client-${CLIENT_JOB_ID}.log" "./client-${CLIENT_JOB_ID}.log"
tcloud download "CSIT6000V-project-results/job-${CLIENT_JOB_ID}.json" "./result-${CLIENT_JOB_ID}.json"
cat "result-${CLIENT_JOB_ID}.json"

# 7. 主动停止 GPU 服务
tcloud cancel -j "$SERVICE_JOB_ID"
```

**成功标准**：Service 日志有 `Application startup complete` 和本次 URL；Client 日志回答 `95`、`北京`（或语义等价正确回答），并输出 `RESULT_FILE=...`；JSON 已下载。

## 本地 Paper Tutor 如何使用它？

课程官方支持的首先是 TACC **内网 CPU Client → GPU Service**。如果 Windows FastAPI 尚无课程批准的连通方式，**不要**把 `http://计算节点:8000/v1` 当作本地可以访问的地址。

在**获准、建立并验证**网络路径后，才在启动 FastAPI 的 Windows PowerShell 中配置：

```powershell
$env:MODEL_PROVIDER = 'openai'
$env:MODEL_BASE_URL = 'http://127.0.0.1:8001/v1'  # 已建立的受控隧道示例端口
$env:MODEL_NAME = 'qwen3.8-27b'
$env:MODEL_TIMEOUT = '240'
```

还要验证 Qwen Chat 正文、JSON mode 和 Streaming。课程 `qwen_client.py` 使用 `chat_template_kwargs={"enable_thinking": false}`；现有项目 OpenAI 客户端并不默认提供此字段，端到端接入可能需要配置适配。详见 [README_TACC.md §2](README_TACC.md)。

## 出问题先检查

- `tcloud: command not found` → 运行位置应是 Linux Bash，检查 SDK `PATH`。
- `Resources` → 作业等待调度，**不要连续重复提交**。
- `Unexpected SDK environment` → 检查是否混用公共 `proxy` 与私有 `mpi-gemm-v1` 文件。
- 没有 `Application startup complete` → 模型尚未完成启动。
- Client 连接失败 → 服务 Job 是否还运行，`server-url.txt` 是否包含本次真实地址与 `/v1`。
- 运行完忘记取消 GPU → `tcloud cancel -j "$SERVICE_JOB_ID"`。
