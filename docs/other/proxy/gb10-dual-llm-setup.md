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
- クライアントPC：VS Code + Claude Code 拡張（**v2.1.129 以降推奨**、後述のモデル自動検出を使う場合）
- 両機で Ollama が 11434 番を使っている場合は、先に止めておく（[vllm_migration.md](../../pr33-38/vllm_migration.md) の Ollama 停止の手順）

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
LiteLLM も同じ機械で動かすため、`--gpu-memory-utilization` は少し低めにする。

```bash
docker run -d --name vllm-qwen --restart unless-stopped \
  --gpus all --ipc=host \
  -p 127.0.0.1:8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  nvcr.io/nvidia/vllm:26.10-py3 \
  vllm serve Qwen/Qwen3-Coder-Next-FP8 \
    --served-model-name qwen3-coder-next \
    --host 0.0.0.0 --port 8000 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --enable-prefix-caching \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.80
```

**注意**
- `-p 127.0.0.1:8000:8000` で、ホストのループバックだけに公開する（LAN からは届かない）。
  コンテナの中は `--host 0.0.0.0` にしないとホストから届かない。
- `-e HF_TOKEN` は付けない（2章）。
- 公式サンプルの `--tensor-parallel-size 2` は**付けない**（1台構成のため）。
- GB10 は CPU と GPU がメモリを共有するため、LiteLLM や OS の分を残して `--gpu-memory-utilization` はやや控えめにする。
- メモリ不足で起動しない場合は `--max-model-len` を 65536 に下げるか、4bit 量子化版に切り替える。
- ログは `docker logs -f vllm-qwen`。`Application startup complete` が出たら起動完了。

---

## 5. GB10-B：gpt-oss-120b の起動

```bash
docker run -d --name vllm-gptoss --restart unless-stopped \
  --gpus all --ipc=host \
  -p 11434:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  nvcr.io/nvidia/vllm:26.10-py3 \
  vllm serve openai/gpt-oss-120b \
    --served-model-name gpt-oss-120b \
    --host 0.0.0.0 --port 8000 \
    --enable-auto-tool-choice \
    --tool-call-parser openai \
    --reasoning-parser openai_gptoss \
    --enable-prefix-caching \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.85
```

**注意**
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
```bash
python3 -m venv ~/litellm-venv
source ~/litellm-venv/bin/activate
pip install 'litellm[proxy]'
```

### 7.2 設定ファイル
同じフォルダの [litellm-config.yaml](litellm-config.yaml) を配置し、GB10-B の `api_base` の IP と `master_key` を環境に合わせて変更する。GB10-A 側の `api_base` は `127.0.0.1:8000` のままでよい（LiteLLM と同じマシンのため）。

| モデル | `api_base` |
|---|---|
| `qwen3-coder-next`（GB10-A） | `http://127.0.0.1:8000/v1` |
| `gpt-oss-120b`（GB10-B） | `http://192.168.10.12:11434/v1` |

### 7.3 起動
```bash
litellm --config litellm-config.yaml --host 0.0.0.0 --port 11434
# トラブル時は --detailed_debug を付ける
```

常駐させる場合は systemd サービス化する（`ExecStart` の末尾も `--port 11434` にする）。
Docker（`ghcr.io/berriai/litellm`）で動かす場合は `--network host` で起動する（127.0.0.1:8000 に届かせるため）。
その場合は 11434 番が ufw の規則どおりに絞られているか、クライアント以外から確かめる。

### 7.4 動作確認（Anthropic形式）
GB10-A またはクライアントPCから実行する。Windows の PowerShell では `curl` が別コマンドの別名になっているため、Git Bash か WSL で実行するか、`curl.exe` と書く（その場合は改行の `\` とクォートを PowerShell 向けに直す）。

```bash
for m in qwen3-coder-next gpt-oss-120b; do
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
  "model": "opusplan",
  "env": {
    "ANTHROPIC_BASE_URL": "http://192.168.10.11:11434",
    "ANTHROPIC_AUTH_TOKEN": "sk-local-change-me",

    "ANTHROPIC_DEFAULT_OPUS_MODEL": "gpt-oss-120b",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "qwen3-coder-next",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "qwen3-coder-next",

    "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY": "1",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"
  }
}
```

| 設定 | 意味 |
|---|---|
| `ANTHROPIC_BASE_URL` | LiteLLM（GB10-A の 11434番）。クライアントの接続先はここだけ |
| `ANTHROPIC_AUTH_TOKEN` | LiteLLM の `master_key` と同じ値（Hugging Face のトークンとは別物） |
| `ANTHROPIC_DEFAULT_OPUS_MODEL` | プルダウンの「Opus」枠 → gpt-oss-120b（GB10-B） |
| `ANTHROPIC_DEFAULT_SONNET_MODEL` | 「Sonnet」枠 → qwen3-coder-next（GB10-A） |
| `ANTHROPIC_DEFAULT_HAIKU_MODEL` | タイトル生成などの裏方処理 → qwen3-coder-next |
| `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY` | LiteLLM の `/v1/models` を読み、実モデル名を `/model` ピッカーに「From gateway」として追加（v2.1.129+） |
| `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC` | テレメトリ等の外部通信を抑止 |
| `"model": "opusplan"` | プランモード＝Opus枠（gpt-oss）、実行＝Sonnet枠（Qwen）に自動切替 |

> Superpowers プラグインや Context7 MCP など既存の設定がある場合は、`env` に追記する形でマージすること。

---

## 9. 使い方：モデルの切り替え

### 方法A：プルダウン / `/model`（手動切替）
- **Opus** を選択 → gpt-oss-120b（GB10-B、設計・レビュー）
- **Sonnet** を選択 → qwen3-coder-next（GB10-A、実装）
- モデル自動検出が有効なら、`gpt-oss-120b` / `qwen3-coder-next` の実名でも選べる

### 方法B：`opusplan`（おすすめ・自動切替）
1. Shift+Tab などでプランモードにし、設計を依頼 → gpt-oss-120b が設計
2. プランを承認して実行 → qwen3-coder-next が実装

---

## 10. 動作確認チェックリスト

- [ ] `vllm --version` が 0.15.0 以上（イメージ `nvcr.io/nvidia/vllm:26.10-py3`）
- [ ] `docker run` に `HF_TOKEN` を付けずに、両モデルのダウンロードと起動が終わる
- [ ] 6章の `curl .../v1/models` で両GB10のモデル名が見える
- [ ] クライアントPCから GB10-B の 11434番に**届かない**（2章）
- [ ] LiteLLM 経由（GB10-A の 11434番）の `/v1/messages` で両モデルが応答する
- [ ] VS Code の Claude Code で `/model` を開き、切り替え候補が表示される
- [ ] Opus 選択時に GB10-B の `docker logs -f vllm-gptoss` にリクエストが来る
- [ ] Sonnet 選択時に GB10-A の `docker logs -f vllm-qwen` にリクエストが来る
- [ ] ファイル読み書き・コマンド実行などのツール呼び出しが成功する

---

## 11. トラブルシューティング

| 症状 | 確認ポイント |
|---|---|
| `model not found` | `--served-model-name`・LiteLLM の `model_name`・Claude Code の設定値が完全一致しているか |
| 接続できない / タイムアウト | 2章のポート一覧どおりか（GB10-A：`sudo ufw status`、`ss -ltnp` で LiteLLM が 11434・vLLM が 127.0.0.1:8000。GB10-B：`docker ps` で `0.0.0.0:11434->8000/tcp`、`sudo iptables -L DOCKER-USER -n`）。GB10-B の 11434番が GB10-A から届くか |
| 11434 番がすでに使われていて起動しない | Ollama が残っていないか（`sudo systemctl stop ollama && sudo systemctl disable ollama`） |
| 401 / 認証エラー（LiteLLM） | `ANTHROPIC_AUTH_TOKEN` と LiteLLM の `master_key` が一致しているか |
| モデル取得で `401` / `gated` | 利用申請が要るモデルを指定している。トークンは使わない方針なので、公開モデル（本書の2つ）を使う |
| ツールが呼ばれず文章だけ返る | vLLM の `--enable-auto-tool-choice` と `--tool-call-parser`（gpt-oss=`openai`、Qwen=`qwen3_coder`）を確認 |
| 長い会話で途中エラー | `--max-model-len` 超過。値を確認し、Claude Code 側の自動圧縮（`/compact`）を早めに使う |
| Qwen が起動しない（OOM） | FP8 → 4bit 版へ変更、または `--max-model-len` / `--gpu-memory-utilization` を下げる。GB10-A は LiteLLM も動くので余白を残す |
| プルダウンに実名が出ない | Claude Code のバージョン（v2.1.129+）と `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1` を確認。出なくても Opus/Sonnet 枠で切替可能 |
| 原因不明 | `litellm ... --detailed_debug` でリクエスト/レスポンスを確認 |

---

## 参考
- vLLM: Claude Code integration — https://docs.vllm.ai/en/latest/serving/integrations/claude_code/
- vLLM Recipes: gpt-oss-120b — https://recipes.vllm.ai/openai/gpt-oss-120b
- Qwen3-Coder-Next（Hugging Face）— https://huggingface.co/Qwen/Qwen3-Coder-Next
- NVIDIA NGC — vLLM コンテナのタグ一覧 — https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm/tags
- LiteLLM: Use Claude Code with Non-Anthropic Models — https://docs.litellm.ai/docs/tutorials/claude_non_anthropic_models
- Claude Code: Model configuration — https://code.claude.com/docs/en/model-config
