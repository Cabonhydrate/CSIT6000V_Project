# CSIT6000V：TACC 上运行 Qwen3.8-27B / vLLM（详细操作手册）

> **适用项目**：Adaptive Agentic RAG / Paper Tutor。  
> **主要依据**：课程提供的 `project-release-1(1).zip`（公共版）与 `project-release-private-vllm(1).zip`（私有版）。  
> **状态**：以下命令经过文件、参数和脚本静态核查；**尚未使用你的 TACC 账号实际提交 GPU 作业**。某些网络路径需课程方许可。  
> **操作系统**：以 Windows 11 + WSL2 Ubuntu（Bash）为例；所有 `tcloud`、`cd "$PROJECT_DIR"`、`bash ...` 命令均在 **WSL2/Linux Bash** 中执行，不是在 Windows PowerShell 中直接执行。

## 0. 五分钟了解：先选哪一个包？

| 路线 | 压缩包 | 用途 | 是否需要自行编译 vLLM | 推荐程度 |
| --- | --- | --- | --- | --- |
| **A：公共环境** | `project-release-1(1).zip` | 使用课程已有 Qwen3.8-27B 服务环境；提交 GPU 服务和 CPU 客户端，验证推理 | **不需要** | **先做这个** |
| B：私有 vLLM | `project-release-private-vllm(1).zip` | 修改 vLLM 的 Python 源码，重新构建并启动私有服务 | **需要**（先提交构建作业） | 仅在课程/实验确实要求修改 vLLM 时做 |
| C：项目 Web UI 接入 | 本项目 `backend/app/agent/model.py` | FastAPI 通过 OpenAI-compatible `/v1/chat/completions` 调用 TACC 的模型 | 不需要新建模型服务；但需要**平台允许的网络路径** | A 验收通过后再做 |

**非常重要的更正**：课程材料使用 **`tcloud submit` + `tuxiv.conf`**，底层通过 Slurm 调度。不要直接把旧集成包里的 `tacc/jobctl.py` / `job_template.sbatch` 当作课程默认启动方式。它们是通用 Slurm 备选模板，**不是本课程公共环境的标准操作**。除非课程明确授权直连 Slurm，否则按本文使用 `tcloud`。

两个包的管理环境也不同，**不要混用脚本**：公共版用 `proxy` + `/mnt/home/public/course-envs/CSIT6000V/project/activate.sh`，私有版用 `mpi-gemm-v1` + `/mnt/home/public/course-envs/CSIT6000V/project-private-vllm/activate.sh`。

### 你最终应拿到的四个结果

1. `SERVICE_JOB_ID`：GPU Serving 作业 ID。
2. `SERVICE_BASE_URL=http://<本次计算节点>:8000/v1`：**从本次服务日志复制**的地址。
3. `CLIENT_JOB_ID`：独立 CPU Client 作业 ID。
4. `result-<CLIENT_JOB_ID>.json`：下载到本地的模型回答、Token usage、请求耗时。

做到这四项，就完成了课程支持的 **TACC 内部 Serving → Client** 验收；**不能自动推断 Windows 后端已经能直接访问 GPU 节点**。

---

## 1. 路线 A：公共环境，逐步部署（推荐）

### A1. 准备目录、终端和客户端 SDK

**在哪执行**：Windows 用来存放两个课程 ZIP；真正使用 SDK 的命令在 WSL2 Ubuntu/Linux 终端。

在 Windows 中先打开 WSL2 Ubuntu：

```powershell
wsl -d Ubuntu-22.04
```

如果你装的发行版不是 `Ubuntu-22.04`，运行 `wsl -l -v` 查看真实名称，或直接打开 Ubuntu 终端。

在 **WSL2 Bash**：

```bash
uname -m                # 课程 SDK Linux 下载面向 x86_64
mkdir -p ~/csit6000v-tacc
cd ~/csit6000v-tacc
```

将 `project-release-1(1).zip` 放到上述目录。下面展示一个 Windows E 盘文件路径的示例；**替换为自己的实际 ZIP 路径**：

```bash
cp "$(wslpath -u 'E:\Downloads\project-release-1(1).zip')" .
unzip 'project-release-1(1).zip'
cd qwen27b-student-release
pwd
ls setup-sdk-account.sh qwen-public-serve/tuxiv.conf qwen-public-client/main.py
```

若 Windows 下载在其他目录，只修改 `wslpath -u` 中的路径，不要手动猜 WSL 盘符路径。

**检查 SDK**（课程明确指定 `tcloud` v0.4.1）：

```bash
command -v tcloud || true
tcloud --help
```

若不存在，先使用课程说明提供的 [官方 Linux x86_64 SDK v0.4.1](https://github.com/turingaicloud/quickstart/releases/download/v0.4.1/linux-amd64-v0.4.1.zip)，解压后为 `tcloud` 赋可执行权限并加到 `PATH`：

```bash
# 假设已下载并解压到 ~/csit6000v-sdk；按实际文件位置调整
find ~/csit6000v-sdk -type f -name tcloud
chmod +x /实际路径/tcloud
export PATH="/包含tcloud程序的目录:$PATH"
tcloud --help
```

> **关于 Windows**：课程包只明确列出了 Linux x86_64 与 macOS Intel SDK。WSL2 使用 Linux ABI 是一种可尝试的路径，但课程是否批准/支持，仍以助教答复为准。不要尝试把 Linux SDK 直接当作 `.exe` 在 PowerShell 里运行。

**预期**：`tcloud --help` 输出 SDK 的命令列表。如果失败先处理 SDK 安装，不进入提交阶段。

### A2. 配置账号与 SSH 私钥（只需首次）

**在哪执行**：WSL2 Bash。本步骤只配置课程 SDK，不是让你手动 SSH 登录 GPU 节点。

```bash
# 目录由你自己创建，放的是与你已注册公钥相匹配的私钥
mkdir -p ~/.ssh
chmod 700 ~/.ssh
chmod 600 ~/.ssh/YOUR_PRIVATE_KEY

tcloud config -u YOUR_TACC_USERNAME -f "$HOME/.ssh/YOUR_PRIVATE_KEY"
tcloud init
tcloud env ls
```

替换 `YOUR_TACC_USERNAME`、`YOUR_PRIVATE_KEY`。若私钥只存在 Windows 文件系统，可以安全复制至 WSL 的 `~/.ssh/`，再 `chmod 600`；**不要上传私钥到项目目录、Git、结果 JSON 或此聊天**。

**预期**：`tcloud env ls` 返回环境清单。若出现 `Failed to get env list` 或初始化错误，依据课程说明联系 TA，不要自己猜测远程 Conda 环境位置。

### A3. 注册公共课程环境（只需首次或换账号时）

**在哪执行**：WSL2 Bash，**必须在公共版包根目录**（有 `setup-sdk-account.sh` 的目录）。

```bash
cd ~/csit6000v-tacc/qwen27b-student-release
PROJECT_DIR=$(pwd -P)
bash setup-sdk-account.sh
```

**原理**：此脚本为 SDK 注册名为 `proxy` 的管理环境；实际运行时 `qwen-public-serve/job.sh` 再加载课程公共 Project 环境：

```text
/mnt/home/public/course-envs/CSIT6000V/common/proxy
/mnt/home/public/course-envs/CSIT6000V/project/activate.sh
```

不要手动安装另一套 vLLM 或覆盖 `.condarc`。脚本会检查并尽量保留现有配置，但如报告配置冲突，应暂停并联系 TA。

**预期**：类似 `Shared proxy environment registered; no packages installed.` 的提示。

### A4. 提交 GPU Serving 作业

**在哪执行**：WSL2 Bash。`PROJECT_DIR` 必须指向公共版根目录；若打开新终端，需要重新设置它。

```bash
cd "$PROJECT_DIR/qwen-public-serve"
tcloud submit
```

**预期**：SDK 输出一个 GPU 服务作业 ID。手动记录后设置变量，例如：

```bash
SERVICE_JOB_ID=替换成刚才的数字

tcloud ps -j "$SERVICE_JOB_ID"
tcloud cat "slurm_log/qwen-public-serve-${SERVICE_JOB_ID}.log"
```

不断使用同一条 `tcloud cat` 查看日志，直到出现：

```text
Application startup complete
SERVICE_BASE_URL=http://<实际计算节点名称>:8000/v1
```

`SERVICE_BASE_URL` 可能比 Ready 提示更早打印；**必须两个条件都满足**才可以提交客户端。

**服务配置由课程包确定**：`tuxiv.conf` 的 `partition=tacc`、`nodes=1`、`ntasks=1`、`cpus-per-task=8`、`gres=gpu:2`、`time=01:00:00`。不要将它误改成自己的 `sbatch --partition` 模板。模型名为 `qwen3.8-27b`，接口端口 8000、上下文约 32K、并发上限 4（课程文档说明）。

### A5. 复制本次服务地址，提交 CPU Client

**在哪执行**：WSL2 Bash。这里最容易出错的是把 `NODE` 原样留下，或误用上一次服务节点地址。

```bash
cd "$PROJECT_DIR"

# 示例！必须替换为本次日志的 SERVICE_BASE_URL 右侧完整地址
SERVICE_BASE_URL='http://替换为本次计算节点:8000/v1'

printf '%s\n' "$SERVICE_BASE_URL" > qwen-public-client/server-url.txt
cat qwen-public-client/server-url.txt

cd "$PROJECT_DIR/qwen-public-client"
tcloud submit
```

**预期**：输出 CPU Client Job ID，再记录并检查：

```bash
CLIENT_JOB_ID=替换成刚才的数字

tcloud ps -j "$CLIENT_JOB_ID"
tcloud cat "slurm_log/qwen-public-client-${CLIENT_JOB_ID}.log"
```

**成功日志应包含**两道默认问题的输出（示例是 `37+58=95` 与“中国的首都=北京”）、Token usage、每个请求耗时，以及：

```text
RESULT_FILE=CSIT6000V-project-results/job-<client-job-id>.json
```

CPU Client 需要保持 GPU Service 在运行；若 `tcloud ps` 显示 `Resources`，说明作业正在等待调度，不是客户端代码错误。

### A6. 下载结果到本地

**在哪执行**：WSL2 Bash，在公共包根目录。

```bash
cd "$PROJECT_DIR"
tcloud download "slurm_log/qwen-public-client-${CLIENT_JOB_ID}.log" "./client-${CLIENT_JOB_ID}.log"
tcloud download "CSIT6000V-project-results/job-${CLIENT_JOB_ID}.json" "./result-${CLIENT_JOB_ID}.json"

cat "result-${CLIENT_JOB_ID}.json"
```

课程脚本把持久结果存入个人 `TACC_USERDIR/CSIT6000V-project-results/`；`tcloud download` 的远端路径以个人 USERDIR 为基准，而不是计算节点的临时 WORKDIR。

**可选的项目实验**：在 `qwen-public-client/main.py` 里替换默认 `prompts` 为 Paper Tutor 的代表性提问，仍通过 `from qwen_client import chat` 调用 `QWEN_BASE_URL`。不要删除 `qwen_client.py`、`job.sh` 或 `tuxiv.conf`，因为它们负责服务地址校验和环境激活。需要保存更多指标时，沿用 `main.py` 的 `records` JSON 结构。

### A7. 结束时取消 GPU Serving 作业（必须做）

```bash
tcloud cancel -j "$SERVICE_JOB_ID"
tcloud ps -j "$SERVICE_JOB_ID"
```

**原因**：GPU 服务会持续占用资源，直至取消或达到 1 小时限额。取消后原来的 `SERVICE_BASE_URL` **不再可用**。下次必须重新 `tcloud submit` GPU 服务、等待 Ready、重新取得节点地址、更新 `server-url.txt`，不能复用旧地址。

### A8. 一次性复制的命令总览

下面是已经配置过 `tcloud` 并完成 `setup-sdk-account.sh` 后的**重复使用流程**（需要手工填 Job ID 和地址）：

```bash
cd ~/csit6000v-tacc/qwen27b-student-release
PROJECT_DIR=$(pwd -P)

# GPU Serving
cd "$PROJECT_DIR/qwen-public-serve"
tcloud submit
SERVICE_JOB_ID=这里填服务作业ID
tcloud cat "slurm_log/qwen-public-serve-${SERVICE_JOB_ID}.log"

# 日志中确认 Application startup complete，然后填写本次服务地址
cd "$PROJECT_DIR"
SERVICE_BASE_URL='http://这里填计算节点:8000/v1'
printf '%s\n' "$SERVICE_BASE_URL" > qwen-public-client/server-url.txt

# CPU Client
cd "$PROJECT_DIR/qwen-public-client"
tcloud submit
CLIENT_JOB_ID=这里填客户端作业ID
tcloud cat "slurm_log/qwen-public-client-${CLIENT_JOB_ID}.log"

# 下载与停止
cd "$PROJECT_DIR"
tcloud download "CSIT6000V-project-results/job-${CLIENT_JOB_ID}.json" "./result-${CLIENT_JOB_ID}.json"
tcloud cancel -j "$SERVICE_JOB_ID"
```

---

## 2. FastAPI / Agent / Web UI 如何接入 TACC？

### 2.1 先理解网络边界

目前课程官方验证链路是：

```text
你的 Windows / WSL2 Bash（tcloud SDK）
            │ 提交作业 + 查看日志
            ▼
TACC GPU Service Job  ── 内部网络 ── TACC CPU Client Job
       vLLM                              qwen_client.py
       /v1/chat/completions              结果存 TACC_USERDIR
```

这条链路**已经在课程文件中提供示例**。它**不代表** Windows 里的 `http://NODE:8000/v1` 一定可访问：`NODE` 是计算节点主机名，外部网络可能不可达，平台规则也可能禁止端口转发。

因此，把整个 Paper Tutor 连接到 TACC 有两个不同验收级别：

- **L3/L4 内部集成（优先）**：用官方 CPU Client Job 调用 Qwen 服务、获得结果及 GPU/作业记录；不必移动本地 PDF、FAISS 索引或前端。
- **端到端 Web UI（条件满足后）**：仅在 TA/平台确认允许的情况下，建立受控连接，让**本地** FastAPI 的 `ModelClient` 调用 TACC 服务。否则保持本地 Ollama UI 作为 Demo，把 TACC 模型实验作为独立证据。

### 2.2 本地后端已有的模型接口

现有 `backend/app/agent/model.py` 支持 OpenAI-compatible `/v1/chat/completions`。若网络**已获允许且经实测可达**，在**启动 FastAPI 的同一个 Windows PowerShell 终端**设置：

```powershell
# 下例假定已建立获准的本机隧道，把 TACC 模型映射到 127.0.0.1:8001
$env:MODEL_PROVIDER = 'openai'
$env:MODEL_BASE_URL = 'http://127.0.0.1:8001/v1'
$env:MODEL_NAME = 'qwen3.8-27b'
$env:MODEL_API_KEY = 'local'  # 仅当服务没有另设密钥；若有密钥使用安全的环境变量
$env:MODEL_TIMEOUT = '240'
$env:MODEL_REASONING_EFFORT = ''

cd backend
uvicorn app.main:app --reload --port 8000 --no-access-log
```

**请注意**：`127.0.0.1:8001` 仅是**经过批准的隧道在本机监听的示例端口**，不是 TACC 的默认地址。课程包并未提供可从 Windows 直接使用的隧道命令或许可。**不要直接对公网开放计算节点的 8000 端口**。若尚未建立合规的连接，不执行上述切换。

验证前先用模型 HTTP 接口检查 `/v1/models` 和 `/v1/chat/completions`；再访问 FastAPI `/api/health`，确认 `model_provider=openai`、模型名称正确、模型可用，最后用论文问答走通一次完整 AgentRun。

### 2.3 Thinking / JSON / Streaming 三项兼容性检查

课程公共版 `qwen-public-client/qwen_client.py` 在 Chat 请求中明确设置：

```python
"chat_template_kwargs": {"enable_thinking": False}
```

这对 Qwen 的**非思考模式、响应正文**很重要。现有 `ModelClient` 的 OpenAI-compatible 通路并不默认传递这个字段，因此不能只因课程默认 CPU Client 成功，就声称整个 Paper Tutor 已完全适配 TACC vLLM。

端到端接入时至少核实：

1. Chat 的 `message.content` 非空（不能只有 reasoning）。
2. `response_format={"type":"json_object"}` 可以正常执行；现有 Agent 的部分工具使用 JSON 输出。
3. Streaming 的 `delta.content` 可被前端/模型客户端正确解析。

如出现“模型已启动但 Agent 不生成正文 / JSON 模式报 400”，先检查服务端的 Qwen thinking 配置及兼容请求格式，再决定是否为 `ModelClient` 加适配；**不要在 README 中把这项兼容写成已经完成**。

### 2.4 回退到本地 Ollama

在新的 Windows PowerShell 终端启动本地服务，或还原当前终端的环境变量：

```powershell
$env:MODEL_PROVIDER = 'ollama'
$env:MODEL_BASE_URL = 'http://127.0.0.1:11434/v1'
$env:MODEL_NAME = 'qwen3:4b'
$env:MODEL_API_KEY = 'local'
$env:MODEL_REASONING_EFFORT = ''
```

然后从同一终端重启 FastAPI。替换生成模型不要求重做本地论文 Embedding，因为向量编码仍在本地完成；但性能和输出质量可能变化，应分别记录模型版本。

---

## 3. 路线 B：私有 vLLM（仅需要改 vLLM 源码时）

> 这不是公共流程的“升级开关”，是**另一套独立包、独立配置和构建流程**。如果你只要用课程 27B 模型做 Agent/RAG，优先跳过这一章。

私有版包的顶层目录是 `qwen27b-private-vllm-student-release`，内含 `vllm-v0.26.0.tar.gz`（课程固定源码版本）、构建作业和独立的 serve/client 作业。

**在 WSL2 Bash**：

```bash
cd ~/csit6000v-tacc
# 先将 project-release-private-vllm(1).zip 放到此目录
unzip 'project-release-private-vllm(1).zip'
cd qwen27b-private-vllm-student-release
PRIVATE_DIR=$(pwd -P)

# 这套包使用 mpi-gemm-v1 管理环境，不是公共版 proxy
bash setup-sdk-account.sh

# 仅首次提取；已有 vllm-src 时不要重复执行
bash prepare-source.sh
```

在本地修改 `vllm-src` 里的 Python 源码之后，先停掉仍在使用旧版私有代码的服务，再提交构建：

```bash
cd "$PRIVATE_DIR"
bash submit-build.sh
BUILD_JOB_ID=填入本次构建作业ID

tcloud ps -j "$BUILD_JOB_ID"
tcloud cat "slurm_log/qwen-private-build-${BUILD_JOB_ID}.log"
```

**必须等到日志出现 `PRIVATE VLLM BUILD PASSED`**，再提交私有 Serving；否则 `qwen-private-serve/job.sh` 会检查到环境未就绪并停止。

```bash
cd "$PRIVATE_DIR/qwen-private-serve"
tcloud submit
SERVICE_JOB_ID=填入私有服务作业ID

tcloud ps -j "$SERVICE_JOB_ID"
tcloud cat "slurm_log/qwen-private-serve-${SERVICE_JOB_ID}.log"
```

等 `Application startup complete` 且拿到新的 `SERVICE_BASE_URL` 后，改写私有版 `server-url.txt`：

```bash
cd "$PRIVATE_DIR"
SERVICE_BASE_URL='http://填入本次私有服务计算节点:8000/v1'
printf '%s\n' "$SERVICE_BASE_URL" > qwen-private-client/server-url.txt

cd "$PRIVATE_DIR/qwen-private-client"
tcloud submit
CLIENT_JOB_ID=填入私有客户端作业ID
tcloud cat "slurm_log/qwen-private-client-${CLIENT_JOB_ID}.log"

cd "$PRIVATE_DIR"
tcloud download "CSIT6000V-project-results/job-${CLIENT_JOB_ID}.json" "./private-result-${CLIENT_JOB_ID}.json"
tcloud cancel -j "$SERVICE_JOB_ID"
```

私有版构建过程会替换个人 USERDIR 中的 `CSIT6000V-private-vllm/vllm-src` 和 `.venv`；不要在服务仍运行时重新构建。构建中断残留 `.build-in-progress` 时，先 `tcloud ps` 确认旧构建作业已结束，再联系 TA 检查恢复，不应盲删目录或手动改共享环境。

**私有版资源**：构建作业 8 CPU / 45 分钟；GPU 服务 2 GPU、8 CPU / 1 小时；客户端 1 CPU / 10 分钟；模型名 `qwen3.8-27b`。这套脚本由课程包固定，不需要自己添加新的 pip / conda 环境。

---

## 4. 常见问题（优先查这一表）

| 现象 | 常见原因 | 建议操作 |
| --- | --- | --- |
| `tcloud: command not found` | SDK 不在 PATH，或使用了 Windows PowerShell 运行 Linux 程序 | 打开 WSL2 Bash；确认 `command -v tcloud`、`tcloud --help`；更新当前 shell PATH |
| `Failed to get env list` | 账号 / 私钥 / 平台初始化问题 | 核对 `tcloud config`、`tcloud init`；仍失败联系 TA |
| `Unexpected SDK environment` | 混用了公共 `proxy` 与私有 `mpi-gemm-v1` 包 | 分开解压，分别使用包自己的 `setup-sdk-account.sh`、`job.sh`、`tuxiv.conf` |
| `tcloud ps` 显示 `Resources` | GPU / CPU 作业排队 | 不要重复提交大量 Job；保留 Job ID 并等调度 |
| `tcloud cat` 找不到日志 | Job 尚未启动、ID 错误、目录名不一致 | 核对 `qwen-public-serve-${SERVICE_JOB_ID}.log` 等实际格式 |
| 有 `SERVICE_BASE_URL` 但请求仍失败 | 模型还在加载、未 Ready、当前服务作业已结束 | 等 `Application startup complete`；用当前 Job 的 URL |
| `server-url.txt` 仍包含 `NODE` | 未替换占位符 | 从服务日志复制完整 URL（保留 `/v1`）覆盖此文件 |
| 请求 `Connection refused` / timeout | Client 先启动、地址过期、服务被取消、跨网段不通 | 保持 GPU Job 运行；检查新地址；客户端只在课程作业内访问 |
| 模型 `400` / `404` | URL `/v1` / 模型名 / 请求格式不匹配 | 核对 `qwen3.8-27b`、接口路径、长度、chat_template_kwargs；检查服务日志 |
| 仅有 reasoning，`content` 为空 | Qwen thinking 设置或输出 Token 预算问题 | 对照课程 `qwen_client.py` 的 `enable_thinking=False`；不要把空 content 当成功 |
| 私有构建提示未就绪 | `PRIVATE VLLM BUILD PASSED` 尚未达到或构建失败 | 看 build 日志；成功后才可 serve；中断恢复联系 TA |
| Client JSON 下载不到 | 作业尚未结束、使用错误 Job ID 或远端路径 | 先检查 `RESULT_FILE=`，再用 `CLIENT_JOB_ID` 下载 USERDIR 下相对路径 |
| Windows 后端无法访问 GPU URL | 计算节点属于内部网络，课程没有授予公网访问 | 先用官方 CPU Client 作业完成内网测试；询问 TA 是否允许特定受控隧道 |

---

## 5. 验收与报告如何记录（防止把模拟当真实 GPU 实验）

建议每次实验保留：

```text
TACC route: public / private
Course SDK: tcloud v0.4.1 (if verified by actual output)
Serving job ID: ...
Serving readiness log: ...
SERVICE_BASE_URL: ... (记录计算节点名即可，不公开任何凭据)
Client job ID: ...
Client log: client-<job-id>.log
Client result: result-<job-id>.json
Model ID: qwen3.8-27b
GPU count: 2 (from course tuxiv.conf; actual allocation still needs job log confirmation)
Status: client completed / HTTP working / model JSON/stream working / backend connected
```

其中 **Client Job 成功** 证明了课程链路上的模型调用；只有现有 `ModelClient` 真实完成一个 AgentRun，才能写 **Paper Tutor Backend connected to TACC**；只有 UI 通过获准的网络连接完成问答，才能写 **End-to-end Web UI connected to TACC**。

> **相关文件**：公共版 `qwen-public-serve/job.sh`、`qwen-public-client/main.py`、`qwen-public-client/qwen_client.py`、两份 `tuxiv.conf`；私有版 `prepare-source.sh`、`submit-build.sh`、`qwen-private-build/job.sh`。本手册只解释和使用这些脚本，没有修改课程原始 ZIP。

