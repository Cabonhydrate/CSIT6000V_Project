<!-- TACC-COURSE-SDK-START -->
## TACC GPU / vLLM 部署（CSIT6000V 课程环境，逐步操作）

> **推荐路线：公共环境 `project-release-1(1).zip`**。课程正式提供 `tcloud` SDK + `tuxiv.conf` + `qwen-public-serve`/`qwen-public-client`，无需自行安装 GPU 驱动或编译 vLLM。  
> **详细版（包括私有 vLLM 构建、报错、回滚）**：[`tacc/README_TACC.md`](tacc/README_TACC.md)。  
> **当前真实状态**：提供了可执行脚本及文档，**尚未在你的 TACC 账号上验证 GPU Job/真实网络连通性**。

### TACC 实际配置与路线选择

| 路线 | 使用场景 | 具体工具 | 资源与约束 |
| --- | --- | --- | --- |
| **公共版（先做）** | 使用已有 Qwen3.8-27B 推理服务并验证模型调用 | `tcloud submit`、`qwen-public-serve`、`qwen-public-client` | GPU 服务：1 节点、2 GPU、8 CPU、1 小时；CPU Client：1 CPU、10 分钟 |
| 私有 vLLM 版 | 自己修改并构建 vLLM 的 Python 源码 | `prepare-source.sh`、`submit-build.sh`、`qwen-private-serve` | 先构建（8 CPU、45 分钟），再启动与公共版相近的 GPU 服务 |
| Windows 本地 Paper Tutor → TACC | 用本地 FastAPI Agent 调用远程 GPU 模型 | OpenAI-compatible ModelClient | **只有网络访问/转发被课程允许且实测可达时才做** |

**必须区分**：课程包通过 `tcloud` 来提交作业，其 `tuxiv.conf` 已指定 Slurm 调度资源。**不要直接用**先前通用的 `tacc/jobctl.py`、`tacc/job_template.sbatch` 取代课程脚本。公共版使用 SDK 环境 `proxy`，私有版使用 `mpi-gemm-v1`，不可交叉复制 `job.sh` / `tuxiv.conf`。

### 1. WSL2 / Linux 准备

用 Windows PowerShell 打开 WSL2 Ubuntu：

```powershell
wsl -d Ubuntu-22.04
```

以下均在 **WSL2 Bash** 执行：

```bash
mkdir -p ~/csit6000v-tacc
cd ~/csit6000v-tacc

# 改为你实际的课程包 ZIP 路径（示例 E:\Downloads）
cp "$(wslpath -u 'E:\Downloads\project-release-1(1).zip')" .
unzip 'project-release-1(1).zip'
cd qwen27b-student-release

# 课程 tcloud SDK v0.4.1 需先安装并加入 PATH
tcloud --help
```

若 `tcloud: command not found`，从课程所附说明的 [Linux x86_64 SDK v0.4.1](https://github.com/turingaicloud/quickstart/releases/download/v0.4.1/linux-amd64-v0.4.1.zip) 安装。在 WSL2 中使用 Linux SDK 是可尝试的路径，但是否受到课程平台正式支持需以 TA 说明为准。切勿把私钥放在项目仓库。

### 2. 首次配置账号与公共环境

```bash
# 用你注册的 TACC 用户名和对应私钥路径替换下面占位符
tcloud config -u YOUR_TACC_USERNAME -f "$HOME/.ssh/YOUR_PRIVATE_KEY"
tcloud init
tcloud env ls

PROJECT_DIR=$(pwd -P)  # 必须在 qwen27b-student-release 根目录
bash setup-sdk-account.sh
```

检查是否成功：`tcloud env ls` 能返回平台环境，脚本提示注册 `proxy`；不要为课程公共运行时自行安装私有 vLLM。若遇到现存 `.condarc` 冲突，按课程 README 联系 TA。

### 3. 提交 2 GPU Serving Job

```bash
cd "$PROJECT_DIR/qwen-public-serve"
tcloud submit

# 从上面命令输出中复制你的实际数字作业 ID
SERVICE_JOB_ID=填写服务作业ID

tcloud ps -j "$SERVICE_JOB_ID"
tcloud cat "slurm_log/qwen-public-serve-${SERVICE_JOB_ID}.log"
```

**必须等到日志出现 `Application startup complete`**。同时找到：

```text
SERVICE_BASE_URL=http://实际计算节点名称:8000/v1
```

这不是固定地址，**每次重新提交服务都要重新复制**。模型名固定为课程文档中的 `qwen3.8-27b`。注意 `Resources` 状态表示等待调度，可能不是报错。

### 4. 复制服务地址并提交 CPU Client Job

```bash
cd "$PROJECT_DIR"
# 把下一行替换成最新服务日志里 SERVICE_BASE_URL= 右侧的完整地址
SERVICE_BASE_URL='http://实际计算节点名称:8000/v1'
printf '%s\n' "$SERVICE_BASE_URL" > qwen-public-client/server-url.txt
cat qwen-public-client/server-url.txt

cd "$PROJECT_DIR/qwen-public-client"
tcloud submit
CLIENT_JOB_ID=填写客户端作业ID

tcloud ps -j "$CLIENT_JOB_ID"
tcloud cat "slurm_log/qwen-public-client-${CLIENT_JOB_ID}.log"
```

成功的 Client 日志通常会回答 `37+58=95` 和“中国的首都=北京”，并显示 Token Usage、每次请求耗时以及结果文件位置：

```text
RESULT_FILE=CSIT6000V-project-results/job-<CLIENT_JOB_ID>.json
```

Client 是**独立 CPU Job**，调用课程 GPU Service，确保 GPU Serving Job 仍在运行。

### 5. 下载运行结果，结束时释放 GPU

```bash
cd "$PROJECT_DIR"
tcloud download "slurm_log/qwen-public-client-${CLIENT_JOB_ID}.log" "./client-${CLIENT_JOB_ID}.log"
tcloud download "CSIT6000V-project-results/job-${CLIENT_JOB_ID}.json" "./result-${CLIENT_JOB_ID}.json"
cat "result-${CLIENT_JOB_ID}.json"

# 测试结束主动释放 GPU
tcloud cancel -j "$SERVICE_JOB_ID"
```

这些命令成功后，才可以在项目报告里写“**TACC 模型服务通过课程内部 CPU Client Job 实测**”。下载结果里的 JSON 包含模型回答、Token usage 和请求耗时。

### 6. 如何接到现有 FastAPI？

Paper Tutor 的 `backend/app/agent/model.py` 已支持 OpenAI-compatible API。**但课程只明确保证平台内部 CPU Client → GPU Service**；Windows 直接访问计算节点是否被允许，需要提前询问 TA，不能把 `SERVICE_BASE_URL` 原样复制到本地后端就当作能连。

若**学校批准并已建立受控隧道**，且模型服务能被该隧道访问，在启动后端的同一个 Windows PowerShell 中设置：

```powershell
$env:MODEL_PROVIDER = 'openai'
$env:MODEL_BASE_URL = 'http://127.0.0.1:8001/v1'  # 经批准的隧道本机示例端口
$env:MODEL_NAME = 'qwen3.8-27b'
$env:MODEL_API_KEY = 'local'  # 如服务要求鉴权，使用实际密钥（不要写进 Git）
$env:MODEL_TIMEOUT = '240'
$env:MODEL_REASONING_EFFORT = ''
cd backend
uvicorn app.main:app --reload --port 8000 --no-access-log
```

**还有一项重要兼容性**：课程 `qwen_client.py` 给 Qwen 的请求带 `chat_template_kwargs={"enable_thinking": false}`；当前 Paper Tutor 的 OpenAI-compatible `ModelClient` 并未默认添加这个字段。必须进一步核对 Chat 非空、JSON mode、Streaming，才能称 Agent 真正接入 TACC；不能只看到 `/v1/models` 返回模型名就算完成。具体检查、私有源码构建、错误处理见 [`tacc/README_TACC.md`](tacc/README_TACC.md)。

**回退本地 Ollama**：

```powershell
$env:MODEL_PROVIDER = 'ollama'
$env:MODEL_BASE_URL = 'http://127.0.0.1:11434/v1'
$env:MODEL_NAME = 'qwen3:4b'
$env:MODEL_API_KEY = 'local'
```

从同一终端重启 FastAPI。TACC 27B 与本地 4B 的结果应分别统计，不能把模型规模差异归因于 RAG 策略本身。

### 7. 关键验收节点

- [ ] `tcloud --help` 成功，`tcloud env ls` 成功。
- [ ] `setup-sdk-account.sh` 注册公共 `proxy` 环境。
- [ ] GPU 服务日志出现 `Application startup complete` 和本次 `SERVICE_BASE_URL`。
- [ ] CPU Client Job 通过，下载 `result-<CLIENT_JOB_ID>.json`。
- [ ] 用 `tcloud cancel -j "$SERVICE_JOB_ID"` 释放 GPU 资源。
- [ ] **可选**：网络获准后由 FastAPI 完成真实模型请求、JSON/Streaming 和一个论文 AgentRun。

<!-- TACC-COURSE-SDK-END -->
