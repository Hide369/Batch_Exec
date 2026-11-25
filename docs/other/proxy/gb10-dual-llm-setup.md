# GB10 ×2 ローカルLLM + Claude Code（VS Code）構築手順書

GB10 を2台用意し、1台目（GB10-A）に `Qwen3-Coder-Next`、2台目（GB10-B）に `gpt-oss-120b` を
vLLM で載せ、LiteLLM で束ねて VS Code の Claude Code からモデルを切り替えて使う手順。

> **この文書の位置づけ（2026-11-20）**
> local-llm-rag リポジトリの `docs/proxy/` から移した。移すときに次を変えた。
> - GB10-A と GB10-B に載せるモデルを入れ替えた（A＝Qwen3-Coder-Next、B＝gpt-oss-120b）。
> - Hugging Face のトークン（`HF_TOKEN`）を使わず、NGC のイメージタグ
>   `nvcr.io/nvidia/vllm:26.10-py3` を指定して vLLM を起動する形にした。
> - ポートを、外部向け **11434**・vLLM 向け **8000** に揃え、開けるポートを2章にまとめた。
>
> **関連文書**:
> - 1台で Ollama を止めて vLLM に切り替える手順（同じイメージタグ）は
>   [vllm_migration.md](../../pr33-38/vllm_migration.md)
> - GB10 1台を Ollama で直接つなぐ手順は local-llm-rag の
>   [VS CodeのClaude CodeをGB10のQwen3-Coder-Nextへ向ける](https://github.com/Hide369/local-llm-rag/blob/master/docs/qwen/claude-code-gb10-qwen3-coder-next.md)、
>   1台構成で LiteLLM を挟んでいた旧手順は [claude-code-gb10.md](https://github.com/Hide369/local-llm-rag/blob/master/docs/claude-code-gb10.md)
> - 推論エンジンを SGLang にする場合は [gb10-dual-sglang-setup.md](gb10-dual-sglang-setup.md)
>
> 本書は **2台をモデル名で振り分ける**ためにゲートウェイを置く構成である。
>
> **検証状況（2026-11-25）**: 移行先の GB10 ×2 で本書の手順（vLLM 版）どおりに構築し、LiteLLM ゲートウェイ経由で
> クライアントの Claude Code からモデルを切り替えて、両モデルが応答することを確かめた。
> 確認した項目は 10章のチェックリストに「2026-11-25 確認済み」と記した。印のない項目はまだ確かめていない。

## 1. 構成概要

```
[VS Code + Claude Code 拡張]
          │  Anthropic Messages API (/v1/messages)
          ▼  http://192.168.10.11:11434
[LiteLLM ゲートウェイ :11434]   ← GB10-A 上で稼働
     │                          │
     ▼ 127.0.0.1:8000           ▼ 192.168.10.12:11434
[GB10-A]                    [GB10-B]
 vLLM :8000                  vLLM（コンテナ内 :8000 → ホスト :11434）
 qwen3-coder-next            gpt-oss-120b
 （実装・コーディング）        （設計・レビュー）
```

| 役割 | ホスト | IP（例） | vLLM の待ち受け | ホスト側で公開するポート | served-model-name |
|---|---|---|---|---|---|
| コーディング用LLM | GB10-A | 127.0.0.1（ループバックのみ） | 8000 | なし（127.0.0.1:8000 のみ） | `qwen3-coder-next` |
| 設計用LLM | GB10-B | 192.168.10.12 | 8000（コンテナ内） | **11434** | `gpt-oss-120b` |
| ゲートウェイ | GB10-A | 192.168.10.11 | – | **11434** | – |

> **ポイント**: Claude Code の接続先（`ANTHROPIC_BASE_URL`）は1つだけ。2台を束ねるために LiteLLM を挟み、モデル名で振り分ける。

---

## 2. 前提条件

- GB10 ×2（DGX OS、NVIDIA ドライバ・Docker・NVIDIA Container Toolkit 導入済み）
- 両機が同一LANにあり、固定IPを割り当て済み
- クライアントPC：VS Code + Claude Code 拡張（**v2.1.242 以降推奨**。プルダウンにモデルの表示名を出す `modelPicker` を使うため。8章）
- 両機で Ollama が 11434 番を使っている場合は、先に止めておく（[vllm_migration.md](../../pr33-38/vllm_migration.md) の Ollama 停止の手順）
- 1台構成の vLLM（[vllm_migration.md](../../pr33-38/vllm_migration.md) の手順1.5で作ったコンテナ `vllm`）が動いている機械は、先に止めておく（4章の注意）

### vLLM のイメージ（トークンは使わない）

- **Hugging Face のトークンは使わない。** Hugging Face にはログインできない前提で、`HF_TOKEN` も
  `hf auth login` も使わない。どちらのモデルも利用申請の要らない公開モデルなので、トークンなしで取得できる。
- vLLM は pip で入れず、NGC のコンテナを**タグを指定して**使う。タグは
  [vllm_migration.md](../../pr33-38/vllm_migration.md) と同じ `nvcr.io/nvidia/vllm:26.10-py3`。
  `latest` は使わない（中身が変わると手順が再現しない）。
- Qwen3-Coder-Next は **vLLM 0.15.0 以上**が必要。3.1 の `vllm --version` で確かめる。

### ネットワーク（開けるポート）

**ファイアウォールで開けるのは 11434/tcp だけ**（両機とも）。8000 番は vLLM がマシンの中で使うだけで、開けない。
4000 番は本構成では使わない。

| サーバー | ポート | 待ち受けるもの | 開ける？ | 接続を許可する相手 |
|---|---|---|---|---|
| GB10-A | **11434/tcp** | LiteLLM ゲートウェイ（外部向け） | **開ける** | クライアントPCのLAN（192.168.10.0/24） |
| GB10-A | 8000/tcp | vLLM（Qwen3-Coder-Next） | 開けない | GB10-A 自身（127.0.0.1 のみで待ち受け） |
| GB10-B | **11434/tcp** | vLLM（gpt-oss-120b。コンテナ内 8000 番へ転送） | **開ける** | GB10-A（192.168.10.11）のみ |
| GB10-B | 8000/tcp | vLLM（コンテナ内） | 開けない | – （ホストには公開しない） |

- クライアントPCが接続するのは **GB10-A の 11434番だけ**。vLLM へクライアントから直接つながない。
- GB10-A の vLLM は LiteLLM が同じマシンから呼ぶので、ループバック（127.0.0.1:8000）だけで待ち受ける。
- GB10-B の vLLM は、コンテナの中では 8000 番で待ち受け、`-p 11434:8000` でホストの 11434 番に出す。
  LAN から見えるのは 11434 番だけで、届いてよいのは GB10-A だけ。
- SSH（22/tcp）など管理用の既存の許可は変えない。
- 外向きは、初回のイメージ取得・モデル取得に **443/tcp**（nvcr.io・huggingface.co）が必要。取得後は閉じてもよい。
- クライアントの外向き通信は `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`（7章）で抑える。ただし Context7 MCP を使う場合、その通信は残る。

GB10-A（LiteLLM は Docker を使わないので ufw で絞れる）：

```bash
sudo ufw allow from 192.168.10.0/24 to any port 11434 proto tcp
sudo ufw status numbered
```

GB10-B（vLLM は Docker の `-p` で公開するので ufw が効かない。`DOCKER-USER` チェーンで絞る）：

```bash
# ホストの 11434 番に来た接続のうち、GB10-A 以外からのものを捨てる
# （DOCKER-USER ではコンテナ側のポート 8000 に書き換わった後なので、元の宛先ポートで判定する）
sudo iptables -I DOCKER-USER -p tcp -m conntrack --ctorigdstport 11434 --ctdir ORIGINAL \
  ! -s 192.168.10.11 -j DROP
sudo apt install -y iptables-persistent && sudo netfilter-persistent save   # 再起動後も残す
```

> 規則を入れたら、クライアントPCから `curl http://192.168.10.12:11434/v1/models` を叩き、
> **届かない**ことを確かめる（GB10-A からは届く。5章）。

---

## 3. GB10 共通の準備（A・B 両方）

### 3.1 イメージ取得とキャッシュ用フォルダ

```bash
docker pull nvcr.io/nvidia/vllm:26.10-py3
docker run --rm nvcr.io/nvidia/vllm:26.10-py3 vllm --version   # 0.15.0 以上であること
mkdir -p ~/.cache/huggingface
```

### 3.2 トークンなしでのモデル取得について

- モデルは初回起動時に `~/.cache/huggingface` へダウンロードされる。2回目以降はキャッシュを使う。
- `You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN...` の警告は、
  未認証のため速度が制限されるという意味で、ダウンロードは続く。
- `401` や `gated` で失敗するモデル（利用申請が要るモデル）は、トークンがないと取れないので使わない。

---

## 4. GB10-A：Qwen3-Coder-Next の起動

Qwen3-Coder-Next は 80B（MoE）。BF16 だと約160GBで GB10 の128GBに載らないため、**FP8 版**を使う。

起動コマンドは、1台構成の GB10 で安定稼働を確認したもの（[vllm_migration.md の手順1.5](../../pr33-38/vllm_migration.md#コンテナで-vllm-を起動する)）に揃える。
変えるのは、LiteLLM と同じ機械に置くための3か所（コンテナ名・公開ポート・`--served-model-name`）だけである。

```bash
# 起動前にページキャッシュを解放する（GB10はCPUとGPUがメモリを共有するため、空きを増やしておく）
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
free -h   # buff/cache が減り available が増えたことを確認

docker run -d \
  --name vllm-qwen \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 127.0.0.1:8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --restart unless-stopped \
  nvcr.io/nvidia/vllm:26.10-py3 \
  python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-Coder-Next-FP8 \
  --served-model-name qwen3-coder-next \
  --gpu-memory-utilization 0.8 \
  --max-model-len 262144 \
  --kv-cache-dtype fp8 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --enable-prefix-caching
```

| 1台構成（vllm_migration.md）との違い | 1台構成 | GB10-A | 理由 |
|---|---|---|---|
| コンテナ名 | `vllm` | `vllm-qwen` | 2台のどちらのコンテナか見分けるため |
| 公開ポート | `-p 11434:8000` | `-p 127.0.0.1:8000:8000` | 11434 番は LiteLLM が使う。vLLM はループバックだけに出す |
| `--served-model-name` | `claude-opus-4-6` | `qwen3-coder-next` | Claude 名（`claude-opus-4-6`）への付け替えは LiteLLM の `model_name`（7.2）で行う |

**注意**
- **1台構成から組み替える場合は、先に既存の `vllm` コンテナを止める。** `--restart unless-stopped` 付きのため、
  止めないと 11434 番を握ったまま再起動のたびに立ち上がり、LiteLLM が起動できず、メモリも取り合う。
  `docker update --restart no vllm && docker stop vllm`（使わないなら `docker rm -f vllm`）。
- `-p 127.0.0.1:8000:8000` で、ホストのループバックだけに公開する（LAN からは届かない）。
  コンテナの中の API サーバーは既定で全アドレスの 8000 番で待ち受けるので、`--host`・`--port` は書かなくてよい。
- `--ipc=host`・`--ulimit` の意味は [vllm_migration.md の手順1.5](../../pr33-38/vllm_migration.md#コンテナで-vllm-を起動する) の表を参照。
- `-e HF_TOKEN` は付けない（2章）。
- 公式サンプルの `--tensor-parallel-size 2` は**付けない**（1台構成のため）。
- LiteLLM（Python のプロセス1つ）は数百MB程度なので、1台構成と同じ `--gpu-memory-utilization 0.8` のまま載る。
  メモリ不足で起動しない場合は `--max-model-len` を 131072 → 65536 の順に下げる（LiteLLM と Claude Code の設定は変えなくてよい）。
- ログは `docker logs -f vllm-qwen`。`Application startup complete` が出たら起動完了。

---

## 5. GB10-B：gpt-oss-120b の起動

`docker run` のオプションと `--enable-prefix-caching` は GB10-A（＝1台構成で稼働中のコマンド）と同じにし、
モデル固有の引数だけを gpt-oss-120b 用にする。

```bash
# 起動前にページキャッシュを解放する（GB10はCPUとGPUがメモリを共有するため、空きを増やしておく）
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
free -h   # buff/cache が減り available が増えたことを確認

docker run -d \
  --name vllm-gptoss \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 11434:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --restart unless-stopped \
  nvcr.io/nvidia/vllm:26.10-py3 \
  python3 -m vllm.entrypoints.openai.api_server \
  --model openai/gpt-oss-120b \
  --served-model-name gpt-oss-120b \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072 \
  --enable-auto-tool-choice \
  --tool-call-parser openai \
  --reasoning-parser openai_gptoss \
  --enable-prefix-caching
```

**注意**
- GB10-B を1台構成から組み替える場合も、先に既存の `vllm` コンテナを止める（4章の注意と同じ。11434 番がぶつかる）。
- この起動コマンドは GB10-B では実測していない。GB10-A と違って `--kv-cache-dtype fp8` は付けない（gpt-oss-120b は KV キャッシュが小さい。[vllm_migration.md の gpt-oss-120b の表](../../pr33-38/vllm_migration.md#gpt-oss-120b-に切り替える手順qwen3-coder-next-が失敗したとき)）。
  メモリ不足で落ちる場合は `--gpu-memory-utilization` を 0.7 に下げる（1台構成の gpt-oss-120b と同じ値）。
- コンテナの中は 8000 番で待ち受け、`-p 11434:8000` でホストの 11434 番に転送する。`-p` は
  `--network host` と一緒には効かないので、`--network host` は付けない。
- 11434 番は GB10-A からだけ届くよう、2章の `DOCKER-USER` の規則を入れておく。
- `-e HF_TOKEN` は付けない（2章）。
- `--served-model-name` は**スラッシュなし**にする（Claude Code は `openai/gpt-oss-120b` のような名前を扱えない）。
- gpt-oss のツール呼び出しは `tool_choice="auto"` のみ対応。
- ログは `docker logs -f vllm-gptoss`。

---

## 6. 各サーバーの動作確認

GB10-A で実行する（GB10-A の 8000番はループバックのみ、GB10-B の 11434番は GB10-A からのみ届く）：

```bash
curl http://127.0.0.1:8000/v1/models        # qwen3-coder-next が見えること（GB10-A 自身）
curl http://192.168.10.12:11434/v1/models   # gpt-oss-120b が見えること（GB10-B）
```

---

## 7. LiteLLM ゲートウェイの構築（GB10-A）

### 7.1 インストール

LiteLLM は**専用の仮想環境**（`~/litellm-venv`）に入れる。GB10 に batch_exec 用の仮想環境があっても、そこには入れない。

- batch_exec の `requirements.txt` は `fastapi`・`uvicorn`・`httpx` などのバージョンを固定している。
  `litellm[proxy]` も同じ系統のライブラリを入れるため、同じ環境だと上書きや依存関係のエラーが起き、Web 画面やバッチの動作が変わるおそれがある。
- LiteLLM は更新が頻繁で、LiteLLM を上げると batch_exec が、`pip install -r requirements.txt` をやり直すと LiteLLM が壊れる関係になる。
- GB10-A で batch_exec を動かしている場合、バッチも 11434 番の LiteLLM 経由で Qwen を呼ぶようになる。環境を分けて、片方の不具合をもう片方に持ち込まないようにする。

```bash
python3 -m venv ~/litellm-venv
~/litellm-venv/bin/pip install 'litellm[proxy]'
~/litellm-venv/bin/litellm --version   # 入ったことを確かめる
```

`source ~/litellm-venv/bin/activate` で有効化してもよいが、batch_exec の仮想環境を有効にしたまま続けて操作すると取り違えやすい。
上のように `~/litellm-venv/bin/` のフルパスで呼べば、どの環境が有効になっていても LiteLLM 専用の環境が使われる。

### 7.2 設定ファイル
同じフォルダの [litellm-config.yaml](litellm-config.yaml) を配置し、GB10-B の `api_base` の IP と `master_key` を環境に合わせて変更する。GB10-A 側の `api_base` は `127.0.0.1:8000` のままでよい（LiteLLM と同じマシンのため）。

| `model_name`（Claude Code が送る名前） | 行き先（vLLM の `--served-model-name`） | `api_base` |
|---|---|---|
| `claude-opus-4-6` | `qwen3-coder-next`（GB10-A） | `http://127.0.0.1:8000/v1` |
| `claude-opus-4-7` | `gpt-oss-120b`（GB10-B） | `http://192.168.10.12:11434/v1` |

**`model_name` は Opus 名だけにし、`claude-sonnet-*` は使わない。** Sonnet 名のモデルでは Claude Code の
自動承認モードが使えないためである。2つのモデルは Opus のバージョン番号（4-6 と 4-7）で区別する。
`claude-opus-4-6` を Qwen3-Coder-Next にしておくと、1台構成（[vllm_migration.md](../../pr33-38/vllm_migration.md)）や
アプリの `LLM_MODEL` と同じ名前で同じモデルに届く。vLLM 側の `--served-model-name` は変えず、名前の付け替えは LiteLLM だけで行う。

### 7.3 起動
```bash
~/litellm-venv/bin/litellm --config litellm-config.yaml --host 0.0.0.0 --port 11434
# トラブル時は --detailed_debug を付ける
```

常駐させる場合は systemd サービス化する。`ExecStart` には LiteLLM 専用の仮想環境の `litellm` をフルパスで書く
（systemd は `~` を展開しないので `/home/<ユーザー名>/...` と書く）。設定ファイルは `/home/<ユーザー名>/litellm/litellm-config.yaml` に置いた例。

```ini
# /etc/systemd/system/litellm.service
[Unit]
Description=LiteLLM gateway (Claude Code -> vLLM)
After=network-online.target docker.service
Wants=network-online.target

[Service]
User=<ユーザー名>
WorkingDirectory=/home/<ユーザー名>/litellm
ExecStart=/home/<ユーザー名>/litellm-venv/bin/litellm --config /home/<ユーザー名>/litellm/litellm-config.yaml --host 0.0.0.0 --port 11434
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now litellm
journalctl -u litellm -f   # 起動ログ
```

LiteLLM を更新するときも、専用の環境だけを触る（`~/litellm-venv/bin/pip install -U 'litellm[proxy]'` → `sudo systemctl restart litellm`）。batch_exec の仮想環境は変わらない。

Docker（`ghcr.io/berriai/litellm`）で動かす場合は `--network host` で起動する（127.0.0.1:8000 に届かせるため）。
その場合は 11434 番が ufw の規則どおりに絞られているか、クライアント以外から確かめる。

### 7.4 動作確認（Anthropic形式）
GB10-A またはクライアントPCから実行する。Windows の PowerShell では `curl` が別コマンドの別名になっているため、Git Bash か WSL で実行するか、`curl.exe` と書く（その場合は改行の `\` とクォートを PowerShell 向けに直す）。

```bash
for m in claude-opus-4-6 claude-opus-4-7; do   # Qwen3-Coder-Next / gpt-oss-120b
  curl -s http://192.168.10.11:11434/v1/messages \
    -H "x-api-key: sk-local-change-me" \
    -H "anthropic-version: 2023-06-01" \
    -H "content-type: application/json" \
    -d "{\"model\":\"$m\",\"max_tokens\":64,\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}"
  echo
done
```
両モデルから応答が返ればOK。

---

## 8. クライアント：Claude Code の設定

VS Code 拡張は `~/.claude/settings.json`（Windows では `%USERPROFILE%\.claude\settings.json`）を読む。以下を設定する。

```json
{
  "model": "claude-opus-4-6",
  "modelPicker": {
    "replaceBuiltInOptions": true,
    "options": [
      {
        "model": "claude-opus-4-6",
        "label": "Qwen3-Coder-Next（実装）",
        "description": "GB10-A / vLLM（claude-opus-4-6）"
      },
      {
        "model": "claude-opus-4-7",
        "label": "gpt-oss-120b（設計・レビュー）",
        "description": "GB10-B / vLLM（claude-opus-4-7）"
      }
    ]
  },
  "env": {
    "ANTHROPIC_BASE_URL": "http://192.168.10.11:11434",
    "ANTHROPIC_AUTH_TOKEN": "sk-local-change-me",

    "ANTHROPIC_DEFAULT_OPUS_MODEL": "claude-opus-4-6",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "claude-opus-4-6",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "claude-opus-4-6",

    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"
  }
}
```

| 設定 | 意味 |
|---|---|
| `"model": "claude-opus-4-6"` | 起動時のモデル → Qwen3-Coder-Next（GB10-A、実装） |
| `modelPicker.options` | プルダウン（`/model`）に出す行。`model` は LiteLLM の `model_name` と完全に一致させる。`label`・`description` は**表示だけ**を変え、送るモデル名（`claude-opus-4-6` など）は変わらない。Claude Code **v2.1.242 以降**が必要 |
| `modelPicker.replaceBuiltInOptions: true` | Claude の既定のモデル行（Opus・Sonnet・Haiku など）を隠し、上の2行と **Default**、使用中のモデルの行だけにする。Sonnet 枠を誤って選ぶこともなくなる |
| `ANTHROPIC_BASE_URL` | LiteLLM（GB10-A の 11434番）。クライアントの接続先はここだけ |
| `ANTHROPIC_AUTH_TOKEN` | LiteLLM の `master_key` と同じ値（Hugging Face のトークンとは別物） |
| `ANTHROPIC_DEFAULT_OPUS_MODEL` | `opus` の割り当て先 → `claude-opus-4-6`（Qwen3-Coder-Next） |
| `ANTHROPIC_DEFAULT_SONNET_MODEL` | `sonnet` の割り当て先。**Default** でもこれが使われる → これも `claude-opus-4-6`。どこを選んでも `claude-sonnet-*` を送らない |
| `ANTHROPIC_DEFAULT_HAIKU_MODEL` | タイトル生成などの裏方処理 → `claude-opus-4-6`（指定しないと存在しない Claude のモデル名を送って失敗する） |
| `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC` | テレメトリ等の外部通信を抑止 |

- **`modelPicker` はユーザー設定（`~/.claude/settings.json`）に書く。** プロジェクトの `.claude/settings.json` や
  `.claude/settings.local.json` に書いても無視される。JSON に書き間違いがあると、その行は捨てられる。
- 以前の版にあった `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY`（LiteLLM の `/v1/models` から行を足す）は外した。
  `replaceBuiltInOptions: true` では、自動検出で見つかった行も隠れるためである。
- VS Code 拡張は、ターミナル版と同じ設定ファイルを読む。ただし、VS Code のプルダウンが `modelPicker` の表示名に従うことは、
  公式ドキュメントに明記されていない。表示が変わらない場合も、`/model` と打てば同じ一覧が出る。

- **Sonnet 名（`claude-sonnet-*`）は使わない。** Sonnet 名のモデルでは自動承認モードが使えないため、
  3つの枠すべてを Opus 名にし、gpt-oss-120b はバージョン違いの `claude-opus-4-7` で呼び分ける。
- 以前の版にあった `"model": "opusplan"`（プランモードは Opus 枠、実行は Sonnet 枠に自動で切り替える）は使わない。
  Sonnet 枠を使う前提の機能のため。設計と実装の切り替えは、9章のとおり手動で行う。

> Superpowers プラグインや Context7 MCP など既存の設定がある場合は、`env` に追記する形でマージすること。

---

## 9. 使い方：モデルの切り替え

チャット入力欄のモデル名のプルダウン（または `/model`）を開くと、8章で付けた表示名の行が出る。

| 選ぶ行 | 送るモデル名 | 使われるモデル | 向いている作業 |
|---|---|---|---|
| Qwen3-Coder-Next（実装） | `claude-opus-4-6` | GB10-A の Qwen3-Coder-Next | 実装、リファクタリング、テスト作成、デバッグ |
| gpt-oss-120b（設計・レビュー） | `claude-opus-4-7` | GB10-B の gpt-oss-120b | 要件整理、設計、方式比較、コードレビュー |

- プルダウンに表示名が出ない場合（Claude Code が v2.1.242 より古いなど）でも、`/model claude-opus-4-7` と
  打てば切り替わる（`claude-opus-4-6` に戻すときも同じ）。
- 設計から実装まで続けて頼むときは、「gpt-oss-120b（設計・レビュー）」で計画を立てさせ、承認したら
  「Qwen3-Coder-Next（実装）」に切り替えて実装させる。会話の履歴は引き継がれる。
- 切り替えた後に自動承認モード（Shift+Tab で切り替えるモード表示）が選べることを確かめる。選べない場合は、
  使っている Claude Code の版がそのモデル名を自動承認モードの対象として扱っていない。次のどちらかを試す（どちらも未検証）。
  - `modelPicker` のその行に `"behavesAs": "claude-opus-4-6"` を足す（v2.1.257 以降）。Claude Code が知らないモデル名に、
    既知のモデルの機能の扱いを当てはめる指定で、送るモデル名と表示名は変わらない
  - `claude-opus-4-7` を別の Opus 名に変える（LiteLLM の `model_name` と `modelPicker` の `model` を一緒に変える）

---

## 10. 動作確認チェックリスト

- [ ] `vllm --version` が 0.15.0 以上（イメージ `nvcr.io/nvidia/vllm:26.10-py3`）
- [ ] `docker run` に `HF_TOKEN` を付けずに、両モデルのダウンロードと起動が終わる
- [ ] 6章の `curl .../v1/models` で両GB10のモデル名が見える
- [ ] クライアントPCから GB10-B の 11434番に**届かない**（2章）
- [ ] LiteLLM 経由（GB10-A の 11434番）の `/v1/messages` で両モデルが応答する（2026-11-25 確認済み）
- [ ] VS Code の Claude Code のプルダウン（または `/model`）に、「Qwen3-Coder-Next（実装）」「gpt-oss-120b（設計・レビュー）」の2行（と Default）だけが出る
- [ ] 「gpt-oss-120b（設計・レビュー）」を選ぶと GB10-B の `docker logs -f vllm-gptoss` にリクエストが来る（2026-11-25 確認済み）
- [ ] 「Qwen3-Coder-Next（実装）」を選ぶと GB10-A の `docker logs -f vllm-qwen` にリクエストが来る（2026-11-25 確認済み）
- [ ] どちらのモデルでも自動承認モードが選べる
- [ ] LiteLLM のログ（`--detailed_debug`）に `claude-sonnet` という名前のリクエストが来ていない
- [ ] ファイル読み書き・コマンド実行などのツール呼び出しが成功する

---

## 11. トラブルシューティング

| 症状 | 確認ポイント |
|---|---|
| `model not found` | Claude Code の設定値（`claude-opus-4-6`・`claude-opus-4-7`）と LiteLLM の `model_name` が完全一致しているか。LiteLLM の `litellm_params.model`（`hosted_vllm/<名前>`）と vLLM の `--served-model-name` が一致しているか |
| `model not found` で名前が `claude-sonnet-...` / `claude-haiku-...` | `ANTHROPIC_DEFAULT_SONNET_MODEL`・`HAIKU_MODEL` が抜けている。8章のとおり `claude-opus-4-6` を入れる |
| 自動承認モードが選べない | 使っているモデル名が Sonnet 名になっていないか（`/status` で確かめる）。9章の最後の注意を参照 |
| 接続できない / タイムアウト | 2章のポート一覧どおりか（GB10-A：`sudo ufw status`、`ss -ltnp` で LiteLLM が 11434・vLLM が 127.0.0.1:8000。GB10-B：`docker ps` で `0.0.0.0:11434->8000/tcp`、`sudo iptables -L DOCKER-USER -n`）。GB10-B の 11434番が GB10-A から届くか |
| 11434 番がすでに使われていて起動しない | Ollama が残っていないか（`sudo systemctl stop ollama && sudo systemctl disable ollama`）。1台構成の `vllm` コンテナが自動起動していないか（`docker ps`。`docker update --restart no vllm && docker stop vllm`） |
| CUDA グラフ取得中（`Capturing CUDA graphs`）に落ちる | CUDA グラフ用のメモリが足りない。`--enforce-eager` を付けて起動し直す（速度は少し下がる） |
| 401 / 認証エラー（LiteLLM） | `ANTHROPIC_AUTH_TOKEN` と LiteLLM の `master_key` が一致しているか |
| モデル取得で `401` / `gated` | 利用申請が要るモデルを指定している。トークンは使わない方針なので、公開モデル（本書の2つ）を使う |
| ツールが呼ばれず文章だけ返る | vLLM の `--enable-auto-tool-choice` と `--tool-call-parser`（gpt-oss=`openai`、Qwen=`qwen3_coder`）を確認 |
| 長い会話で途中エラー | `--max-model-len` 超過。値を確認し、Claude Code 側の自動圧縮（`/compact`）を早めに使う |
| Qwen が起動しない（OOM） | FP8 → 4bit 版へ変更、または `--max-model-len` / `--gpu-memory-utilization` を下げる。GB10-A は LiteLLM も動くので余白を残す |
| プルダウンに表示名の行が出ない / Claude のモデル行が残る | Claude Code が v2.1.242 以降か。`modelPicker` を `~/.claude/settings.json`（ユーザー設定）に書いたか（プロジェクト設定では無視される）。JSON の書き間違いがないか。出なくても `/model claude-opus-4-7` で切替可能 |
| 原因不明 | `litellm ... --detailed_debug` でリクエスト/レスポンスを確認 |

---

## 参考
- vLLM: Claude Code integration — https://docs.vllm.ai/en/latest/serving/integrations/claude_code/
- vLLM Recipes: gpt-oss-120b — https://recipes.vllm.ai/openai/gpt-oss-120b
- Qwen3-Coder-Next（Hugging Face）— https://huggingface.co/Qwen/Qwen3-Coder-Next
- NVIDIA NGC — vLLM コンテナのタグ一覧 — https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm/tags
- LiteLLM: Use Claude Code with Non-Anthropic Models — https://docs.litellm.ai/docs/tutorials/claude_non_anthropic_models
- Claude Code: Model configuration — https://code.claude.com/docs/en/model-config
