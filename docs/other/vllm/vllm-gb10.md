# GB10サーバーにvLLMを立てる

Nvidia GB10（DGX Spark / ASUS Ascent GX10 などのOEM機を含む）を推論サーバーにし、
RAGの生成とコーディングエージェントを社内ネットワークの中で完結させるための手順。

> **どこまで採るかが用途で分かれている（2026-09-16 時点）。**
>
> - **RAG本体（`rag_chat_app.py` の検索と生成）は Ollama のまま。** 接続先を
>   vLLM へ切り替える実装は、取り込まずに閉じた
>   [PR #75](https://github.com/Hide369/local-llm-rag/pull/75) に残っている。
>   再開するならそこから拾える
> - **コーディングエージェントでは vLLM を使う。** 手順は
>   [GB10のvLLM + gpt-oss-120b でCodexとClaude Codeを使う](vllm-gb10-coding-agent.md)
>
> この文書はvLLMサーバー自体の立て方（コンテナ、systemd、公開と認証）で、
> どちらの用途でも土台になる。**実機での起動と実測はしていない。**

> **この文書の位置づけ（2026-11-20）**
> local-llm-rag リポジトリの `docs/vllm/` から移した。移すときに次を変えた。
> - **Hugging Face のトークン（`HF_TOKEN`）は使わない。** ログインできない前提で、
>   `HF_TOKEN` も `hf auth login` も使わず、利用申請の要らない公開モデルだけを対象にする。
>   vLLM は NGC のコンテナを**タグを指定して**使う。タグは
>   [vllm_migration.md](../../pr33-38/vllm_migration.md) と同じ `nvcr.io/nvidia/vllm:26.10-py3`。
> - **ポートを外部向け 11434、vLLM 向け 8000 に揃えた。** ファイアウォールで開いているのは
>   11434 と 4000 だけなので、開けるポートと使い分けを[ポートの決まり](#ポートの決まり)にまとめた。
>
> 文中の `ingest/...` や `scripts/...`、`.env` は local-llm-rag リポジトリのものである。

> **この文書の位置づけ**
> 手順とコマンドは NVIDIA の DGX Spark playbook（`nvidia/vllm`、2026-06-12 更新）と
> vLLMの公開情報に基づく。**手元にGB10が無いため実測はしていない**。
> コンテナのタグ、モデルのHFハンドル、フラグ名は版で変わる。導入時に
> [出典](#出典) の各ページで必ず現物を確認すること。所要時間・トークン毎秒の
> 数値をこの文書に書いていないのは、測っていないものを書かないためである。

> **このリポジトリのサーバーで稼働中の vLLM（2026-11-21）**
> 移行先の GB10 では、[vllm_migration.md の手順1.5](../../pr33-38/vllm_migration.md#コンテナで-vllm-を起動する) の
> コマンドで vLLM がコンテナ名 `vllm` として常駐している（Qwen3-Coder-Next、`--served-model-name claude-opus-4-6`、
> `-p 11434:8000`、`--restart unless-stopped`）。この文書の起動例を同じ GB10 で試すときは、先に
> `docker update --restart no vllm && docker stop vllm` で止める。止めないと 11434 番とメモリを取り合い、
> OS を再起動するたびに `vllm` が先に立ち上がる。戻すときは `docker update --restart unless-stopped vllm && docker start vllm`。
>
> 1台構成で安定稼働を確認した指定のうち、`--ulimit memlock=-1 --ulimit stack=67108864` は
> この文書の起動例に入っていない。起動例が起動直後に落ちるときは、まずこれを足して試す。

## 目次

- [GB10で先に知っておくこと](#gb10で先に知っておくこと)
- [ポートの決まり](#ポートの決まり)
- [1. 前提の確認](#1-前提の確認)
- [2. vLLMコンテナを取得して起動する](#2-vllmコンテナを取得して起動する)
- [3. 常時稼働させる（systemd）](#3-常時稼働させるsystemd)
- [4. 社内ネットワークへの公開と認証](#4-社内ネットワークへの公開と認証)
- [5. モデルの選び方と128GBの割り振り](#5-モデルの選び方と128gbの割り振り)
- [6. コーディングエージェント向けの起動](#6-コーディングエージェント向けの起動)
- [7. 埋め込みとリランカーをvLLMで出す](#7-埋め込みとリランカーをvllmで出す)
- [8. local-llm-ragのRAGとつなぐ](#8-local-llm-ragのragとつなぐ)
- [9. トラブルシュート](#9-トラブルシュート)
- [出典](#出典)

## GB10で先に知っておくこと

普通のx86＋RTXサーバーと違う点が3つあり、これを知らないまま始めると最初の1日を
ビルドエラーで溶かす。

**1. aarch64 かつ sm_121 である。** GB10のGPUはBlackwellの `sm_121`（`sm_121a`）で、
これをサポートするのは **CUDA 13.0以降**である。CPU側はArm（`aarch64`）。
PyPIに並ぶ機械学習系のwheelの多くはいまだにCUDA 12.x・x86_64向けにビルドされて
おり、`pip install vllm` がそのまま通る保証はない。素のpip導入を試すと、PyTorchや
Tritonのカーネルが `sm_120` 止まりで、起動時のCUDAカーネル初期化で落ちる、という
既知の壁に当たる（vLLM issue #36821）。**コンテナから入るのが最短で、かつ再現性がある。**
LLVM/Tritonにパッチを当ててソースからビルドする道もあるが、最初の1台では選ぶ理由がない。

**2. ユニファイドメモリ（UMA）である。** 128GBのLPDDR5XをCPUとGPUが共有する。
専用VRAMを持つGPUと違い、`--gpu-memory-utilization 0.8` は「128GBの80%」を意味する。
OSとページキャッシュも同じ物理メモリに乗るため、**複数のvLLMプロセスを常駐させる
なら、各プロセスの `--gpu-memory-utilization` の合計を管理するのは自分の仕事になる**。
足し算を間違えるとOOMする。

**3. メモリ帯域が律速する。** GB10のメモリ帯域は約273GB/sである。データセンター向けの
HBM搭載GPU（数TB/s）と比べると1桁低い。生成（decode）は1トークンごとに重みを読み直す
処理なので、**1本のストリームの応答速度は帯域でほぼ決まる**。一方でプロンプト処理
（prefill）と多重実行はGB10の演算性能が効く。したがって、

- 「1人が短い質問を投げて速く返ってくる」用途では、専用GPU機ほどの体感は出ない
- 「複数人が同時に使う」「RAGで長い文脈を読ませる」「エージェントが裏で回る」
  用途では、まとめて処理できるvLLMの利点が出る

という傾向になる。MoE（Mixture of Experts）モデルは1トークンあたりに読む重みが
少ないため、この機械では密（dense）モデルより有利である。NVIDIAが提示する推奨構成に
MoEが多いのはこのためである。

## ポートの決まり

**GB10のファイアウォールで開いているのは 11434/tcp と 4000/tcp だけである。** この
フォルダの文書は、すべて次の決まりで書いてある。

|ポート|何が待ち受けるか|開ける？|誰から|
|---|---|---|---|
|**11434/tcp**|外部向けのvLLM（主に使う1本）。**コンテナの中は 8000 番**で、`-p 11434:8000` でホストへ出す|**開ける**|社内LAN（クライアントPC）|
|**4000/tcp**|LiteLLM（Claude Code 用の変換。[vllm-gb10-coding-agent.md](vllm-gb10-coding-agent.md) の3節）|**開ける**（使う場合だけ）|社内LAN（クライアントPC）|
|8000/tcp|vLLM のコンテナの中の待ち受け（`--port 8000`）。どのコンテナでも同じ|開けない|–（ホストには直接出さない）|
|127.0.0.1:8001〜|2本目以降のvLLM（埋め込み・VLM・リランカーなど）。コンテナの中は同じく 8000 番|開けない|GB10 自身だけ|

- **`vllm serve` には常に `--host 0.0.0.0 --port 8000` を渡す。** ホスト側の番号は `-p` だけで決める。
  1台構成で稼働中の `python3 -m vllm.entrypoints.openai.api_server`（[vllm_migration.md の手順1.5](../../pr33-38/vllm_migration.md#コンテナで-vllm-を起動する)）は、
  既定で全アドレスの 8000 番で待ち受けるので、`--host`・`--port` を書かなくても同じになる。
- **LANから届くvLLMは1本だけ**（11434番）。2本目以降をクライアントPCから使いたいときは、
  ファイアウォールに穴を足さず、4000番の LiteLLM に `model_name` で並べて振り分ける。
- **Ollama と同じ 11434 番を使う。** 同じGB10で Ollama を 11434 番のまま動かしているなら、
  先に止める（[vllm_migration.md](../../pr33-38/vllm_migration.md) の Ollama 停止の手順）。
  [8節](#8-local-llm-ragのragとつなぐ)の道(a)のように Ollama を残して外から使うなら、
  11434 番は Ollama に充て、vLLM は `-p 127.0.0.1:8001:8000` でループバックに置き、
  4000番の LiteLLM 経由で外へ出す。
- Docker の `-p` で公開したポートには **ufw が効かない**。送信元の絞り方は[4節](#4-社内ネットワークへの公開と認証)。

## 1. 前提の確認

### OSと、手元のWindowsとの関係

**GB10はWSLではない。素のLinuxが動く独立したマシンである。** DGX Sparkには
DGX OS 7.x（Ubuntu 24.04 LTSベース、ARM64/`aarch64`）が入っている。Windowsは
動かないので、WSLという概念自体がない。OpenSSHが既定で有効なので、初回は
`ssh <ユーザー名>@spark-xxxx.local` で入り、テンポラリパスワードを変更する。
以降は普通のUbuntuサーバーとして扱ってよい。

**手元のWindows PCは今のまま変えない。** local-llm-ragはPowerShellと
`myvenv313` で動き、サーバーに対してやることはHTTPを投げるだけである。

```text
Windows PC（PowerShell + myvenv313 + Streamlit）   ← 現状のまま。WSLは要らない
        │ HTTP/HTTPS（社内LAN）
        ▼
GB10（DGX OS 7 = Ubuntu 24.04, aarch64）
        └── Docker → vLLM ／ Ollama
```

クライアント側にWSLを挟む理由は無い。むしろWSL↔Windows間のネットワークを
1段増やすことになり、READMEに実測が載っている `localhost` がIPv6の `::1` に
先に解決されて1リクエストあたり約2.1秒を浪費する類の問題を、自分から
呼び込むことになる。

WSLが要るのは別の場面である。GB10ではなく**Windowsのx86機にGPUを挿して
vLLMを動かす**なら、vLLMにWindowsネイティブの対応が無いのでWSL2が唯一の道に
なる。今回の構成では該当しない。

### GB10側で確認すること

DGX OSが入った状態から始める。

```bash
# GPUが見えるか。ドライバのバージョンも控えておく
nvidia-smi

# CUDA 13.0 以上であること
nvcc --version

# aarch64 であること（この文書はaarch64前提）
uname -m

# Docker と NVIDIA Container Toolkit
docker --version
docker run --rm --gpus all nvcr.io/nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi
```

最後のコマンドが `nvidia-smi` の出力を返せば、コンテナからGPUが見えている。
権限エラー（`permission denied ... docker.sock`）が出る場合は、sudoを付けるか
dockerグループに入る。

```bash
sudo usermod -aG docker $USER
newgrp docker
```

**HuggingFaceのトークンは使わない。** ログインできない前提なので、`HF_TOKEN` も
`hf auth login` も使わない。トークンが要るのはゲート付き（gated）のモデル（Llama系など）
だけで、公開モデルは匿名で取れる。この文書が扱うのは**ゲートの無い公開モデルだけ**である。
判定と、HFを使わない経路は[入手経路とオフライン運用](#入手経路とオフライン運用)に書いた。

## 2. vLLMコンテナを取得して起動する

NVIDIAがNGCでGB10向けにビルド済みのvLLMコンテナを配っている。これを**タグを指定して**使う。
pip では入れない（[GB10で先に知っておくこと](#gb10で先に知っておくこと)の1）。

```bash
# タグは NGC のタグ一覧（https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm/tags）から
# 取得した 26.10-py3 に固定する（vllm_migration.md と同じ）。latest は使わない
export VLLM_IMAGE=nvcr.io/nvidia/vllm:26.10-py3
docker pull ${VLLM_IMAGE}
docker run --rm ${VLLM_IMAGE} vllm --version     # 入っているvLLMの版を控えておく
```

起動する。**HuggingFaceのキャッシュを必ずホストにマウントすること**。これを忘れると
コンテナを作り直すたびに数十GBのモデルを落とし直すことになる。`-e HF_TOKEN` は付けない。

```bash
mkdir -p ~/.cache/huggingface

docker run -d --name vllm-chat --restart unless-stopped \
  --gpus all --ipc=host \
  -p 127.0.0.1:11434:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ${VLLM_IMAGE} \
  vllm serve openai/gpt-oss-20b \
    --host 0.0.0.0 --port 8000 \
    --gpu-memory-utilization 0.3 \
    --max-model-len 65536 \
    --max-num-seqs 8 \
    --enable-prefix-caching
```

意図を補足する。

|フラグ|なぜそうするか|
|---|---|
|`-p 127.0.0.1:11434:8000`|コンテナの中の 8000 番を、ホストの 11434 番（外部向けの番号）へ出す。素の `-p 11434:8000` は全インターフェースに開くので、まずループバックだけに閉じ、LANへの公開は[4節](#4-社内ネットワークへの公開と認証)で意識的に行う|
|`--host 0.0.0.0 --port 8000`|コンテナの中の待ち受け。`0.0.0.0` にしないとホストから届かない。番号は常に 8000（[ポートの決まり](#ポートの決まり)）|
|`--ipc=host`|共有メモリ不足によるワーカーのクラッシュを避ける（`--shm-size=16g` でも可）|
|`--gpu-memory-utilization 0.3`|128GBの30%＝約38GB。複数モデルを常駐させる前提で最初から絞る。1モデルだけなら0.8まで上げてよい|
|`--max-model-len`|KVキャッシュの量を決める。長すぎる指定はそれだけでメモリを食う|
|`--max-num-seqs`|同時に処理するリクエスト数の上限。増やすほどスループットは上がるが1本あたりは遅くなる|
|`--enable-prefix-caching`|RAGのように毎回同じ長いプロンプト前半を送る用途で効く|
|`--restart unless-stopped`|再起動後も自動で上がる。systemdで管理するなら[3節](#3-常時稼働させるsystemd)|

起動を待つ。初回はモデルのダウンロードとカーネルのコンパイルがあるので数分から
十数分かかる。

```bash
timeout 1800 bash -c 'until curl -sf http://localhost:11434/health > /dev/null 2>&1; do sleep 10; done' \
  || { echo "起動しなかった"; docker logs vllm-chat | tail -50; }
```

疎通を確認する。

```bash
curl http://localhost:11434/v1/models

curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-20b",
    "messages": [{"role": "user", "content": "12*17"}],
    "max_tokens": 128
  }'
```

`204` が含まれる応答が返れば、推論サーバーとして動いている。

> モデルによっては専用のコンテナと起動フラグが要る。Gemma 4系は
> `vllm/vllm-openai:gemma4-cu130`、DiffusionGemmaは `vllm/vllm-openai:gemma` を使う、
> といった指定がplaybookにある。新しいモデルを載せる前に、そのモデルの
> HFモデルカードとplaybookを見ること。別のイメージを使うときも、`latest` ではなく
> このようにタグを指定し、`-e HF_TOKEN` は付けない。

## 3. 常時稼働させる（systemd）

`--restart unless-stopped` だけでも再起動には追随するが、ログと起動順序を
OSの流儀に乗せたいなら、コンテナをsystemdから起こす。

`/etc/systemd/system/vllm-chat.service`:

```ini
[Unit]
Description=vLLM (chat) on GB10
After=docker.service network-online.target
Requires=docker.service

[Service]
Restart=always
RestartSec=10
# コンテナ自身の再起動ポリシーとは二重にしない
ExecStartPre=-/usr/bin/docker rm -f vllm-chat
ExecStart=/usr/bin/docker run --rm --name vllm-chat \
  --gpus all --ipc=host \
  -p 127.0.0.1:11434:8000 \
  --env-file /etc/vllm/chat.env \
  -v /home/<ユーザー>/.cache/huggingface:/root/.cache/huggingface \
  nvcr.io/nvidia/vllm:26.10-py3 \
  vllm serve openai/gpt-oss-20b \
    --host 0.0.0.0 --port 8000 \
    --gpu-memory-utilization 0.3 --max-model-len 65536 \
    --max-num-seqs 8 --enable-prefix-caching
ExecStop=/usr/bin/docker stop vllm-chat
# モデルのロードに時間がかかるので待つ
TimeoutStartSec=1800

[Install]
WantedBy=multi-user.target
```

（使うなら）`VLLM_API_KEY` は `/etc/vllm/chat.env` に置き、`chmod 600` にする。
unitファイルに直接書かない。`HF_TOKEN` は書かない（トークンは使わない）。
LANへ公開するときは、`-p 127.0.0.1:11434:8000` を[4節](#4-社内ネットワークへの公開と認証)の形に替える。

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now vllm-chat
journalctl -u vllm-chat -f
```

モデルを複数常駐させるなら、このunitを名前とホスト側のポートだけ変えて複製する
（`vllm-code.service` → `-p 127.0.0.1:8001:8000`、`vllm-embed.service` → `-p 127.0.0.1:8002:8000` など）。
コンテナの中は 8000 番のままでよい。2本目以降はLANに出さない（[ポートの決まり](#ポートの決まり)）。

## 4. 社内ネットワークへの公開と認証

**vLLMは既定で無認証である。** `--api-key` を付けない限り、到達できる全員が
モデルを使えるし、投げたプロンプトも自由である。社内の他PCから使う時点で、
最低限これを行う。

```bash
# 起動フラグに追加する（VLLM_API_KEY 環境変数でも指定できる）
--api-key "<十分に長いランダム文字列>"
```

クライアントは `Authorization: Bearer <キー>` を付ける。

**LANへ出すのは 11434 番だけである**（[ポートの決まり](#ポートの決まり)）。443 番などは
開いていないので、出し方は次の2つのどちらかになる。

**(1) vLLM を 11434 番で直接出す（簡単）**

2節・3節の `-p 127.0.0.1:11434:8000` を `-p 11434:8000` に替え、`--api-key` を付けて
起動し直す。Docker の `-p` で公開したポートには ufw が効かないので、送信元は
`DOCKER-USER` チェーンで社内LANに絞る。

```bash
# ホストの 11434 番に来た接続のうち、社内LAN（例: 192.168.1.0/24）以外からのものを捨てる。
# DOCKER-USER ではコンテナ側の 8000 番に書き換わった後なので、元の宛先ポートで判定する
sudo iptables -I DOCKER-USER -p tcp -m conntrack --ctorigdstport 11434 --ctdir ORIGINAL \
  ! -s 192.168.1.0/24 -j DROP
sudo apt install -y iptables-persistent && sudo netfilter-persistent save   # 再起動後も残す
```

**(2) TLS終端のリバースプロキシを 11434 番に置く**

推論サーバーを直接LANに晒したくないなら、local-llm-ragの
[docs/server-deployment.md](https://github.com/Hide369/local-llm-rag/blob/master/docs/server-deployment.md) と同じ考え方で、
TLSと認証を担うNginxを前に置く。この場合は Nginx が 11434 番を受け持つので、vLLM は
`-p 127.0.0.1:8000:8000` でループバックに置き直す。

```text
利用者のPC（Streamlit RAG / Codex CLI）
        │ HTTPS :11434
        ▼
Nginx（TLS終端・認証）             ← GB10上。ufw で 11434 番を社内LANにだけ開ける
        │ HTTP 127.0.0.1:8000
        ▼
vLLM（コンテナの中も 8000 番）
```

Nginx はDockerを使わずに動かすなら ufw で絞れる。

```bash
sudo ufw allow from 192.168.1.0/24 to any port 11434 proto tcp
```

どちらでも、**8000番・8001番以降はファイアウォールで開けない**。GB10を持ち出す運用
（会議室に持って行くなど）があるなら、なおさら `-p` を `127.0.0.1:` なしで書かないこと。
規則を入れたら、社内LANの外にあたる端末から 11434 番に届かないことを確かめる。

## 5. モデルの選び方と128GBの割り振り

NVIDIAがGB10での動作を確認しているモデルは playbook の Model Support Matrix に
一覧がある。用途別に、そこから選ぶのが安全である。

> Nemotron系を軸に、生成・VLM・埋め込みの3本を具体的に選定して載せる手順は
> [vllm-gb10-models.md](vllm-gb10-models.md) に分けてある。下の表は
> 全体像をつかむための一覧である。

|用途|候補|量子化|備考|
|---|---|---|---|
|RAGの回答生成|`openai/gpt-oss-20b`|MXFP4|いまlocal-llm-ragがOllamaで使っているものと同じ系列。移行の比較対象にしやすい|
|RAGの回答生成（大きく）|`openai/gpt-oss-120b`|MXFP4|128GBならMoEの120Bが載る。GB10を買った意味が最も出る枠|
|同上|`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`|NVFP4|NVIDIA製。NVFP4はBlackwellの4bit形式で、帯域の節約になる|
|コーディングエージェント|`nvidia/Qwen3.6-35B-A3B-NVFP4`|NVFP4|playbookに「Agent Ready」の起動レシピがある（[6節](#6-コーディングエージェント向けの起動)）|
|図表の説明文化（VLM）|`nvidia/Qwen2.5-VL-7B-Instruct-NVFP4`|NVFP4|local-llm-ragが `qwen3-vl:8b` で行っている処理に対応する|
|埋め込み|`BAAI/bge-m3`|—|local-llm-ragの既定と同じモデル（[7節](#7-埋め込みとリランカーをvllmで出す)、ただし注意あり）|
|リランカー|`BAAI/bge-reranker-v2-m3`|—|同上。いまはCPUのONNXで動いており、移す必然性は薄い|

**128GBの割り振りを先に決める。** vLLMは起動時に `--gpu-memory-utilization` の分だけ
確保して抱え込む。あとから起動するプロセスのために空きを残すのは自分の責任である。
たとえば3プロセス常駐なら、

|プロセス|用途|`--gpu-memory-utilization`|概算|
|---|---|---|---|
|`vllm-chat`|RAGの回答生成|0.35|約45GB|
|`vllm-code`|コーディングエージェント|0.35|約45GB|
|`vllm-embed`|埋め込み|0.05|約6GB|
|（残り）|OS・ページキャッシュ・他プロセス|—|約30GB|

のように配分する。合計を0.9近くまで詰めるとOSごと不安定になる。**最初は1プロセスで
立ち上げ、実際に使う構成が決まってから分割する**のが手戻りが少ない。

量子化の選び方は単純で、**NVFP4版が公開されているならそれを使う**。Blackwellの
FP4は演算器で直接扱え、重みが小さくなるぶん[帯域の律速](#gb10で先に知っておくこと)も
緩む。gpt-ossは配布時点でMXFP4なのでそのままでよい。BF16のチェックポイントは、
比較検証の用途を除けばGB10で選ぶ理由が薄い。

### 入手経路とオフライン運用

この文書のモデルハンドルはすべてHuggingFaceのものだが、**HFが唯一の経路では
ない**。取ってくる物ごとに出どころが違う。

|取ってくる物|出どころ|コマンド|
|---|---|---|
|コンテナイメージ|NGC（`nvcr.io/nvidia/vllm`）|`docker pull`。**HFではない**|
|モデルの重み|HuggingFace|`hf download`。**トークンは使わない**。ゲート付きでない公開モデルだけを取る（下記）|
|同上（代替）|NGC / build.nvidia.com|NVIDIA製モデル（Nemotron系）はNGCにも置かれている|
|Ollamaを併用する場合のモデル|ollama.com のレジストリ|`ollama pull`。HFを経由しない別チャネル|

**閉域・オフラインで運用するなら**、外に出られるマシンで先に取り、
`~/.cache/huggingface` ごと搬入する。

```bash
# 接続できる環境で一度だけ取る
hf download nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4

# GB10へ ~/.cache/huggingface を運んだあと、取りに行かせない
export HF_HUB_OFFLINE=1
```

`HF_HUB_OFFLINE=1` を付けておくと、キャッシュに無いときに黙って外へ出ず、
エラーで落ちる。**閉域では「静かに通信すること」のほうが事故**なので、
これは付けておくほうがよい。`HF_HOME` でキャッシュの場所を変えているなら、
systemdのunitとdockerの `-v` の両方を同じ場所に揃えること。

同じ話がlocal-llm-rag側にもある。`AGENTS.md` に書いてあるとおり、リランカーの
初回チェック（`ingest/reranker.py`）が huggingface.co へHEADを投げる。GB10の
構成とは独立に、オフライン運用では先にモデルを置いておく必要がある。

### HuggingFaceにログインせずに取る

**この文書ではログインもトークンも使わない。公開リポジトリは匿名で落とせる。** アカウントと
トークンが要るのは「ゲート付き（gated）」のリポジトリだけで、代表例が
`meta-llama/*` である。playbookが `HF_TOKEN` を前提に書いているのは、例として
Llama 3.3 70Bを使っているからで、この文書が選んだモデルには当てはまらない
可能性が高い。**ゲート付きのモデルは使わない**（トークンが無いと取れないため）。

**まず、そのモデルがゲート付きかを確かめる。** 落とす前に1行で判定できる。

```bash
curl -s https://huggingface.co/api/models/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print('gated:', d.get('gated'), '/ license:', d.get('cardData',{}).get('license'))"
```

`gated: False` ならログイン不要でそのまま落とせる。`auto` か `manual` が返った
ら、そのリポジトリはブラウザーでの同意（`manual` なら著者の承認まで）とトークンが要るので、
**選ばずに別の公開モデルにする**。

**匿名で落とす。**

```bash
unset HF_TOKEN            # 空のトークンが混ざると逆に401になる
hf download nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
```

`-e HF_TOKEN=...` を渡さずにコンテナを起動すれば、vLLMが起動時に取りに行く分も
同じく匿名で通る。

ひとつだけ注意がある。**匿名のダウンロードはレート制限が厳しい。** 60GB級の
チェックポイントでは途中で絞られることがある。`hf download` は再開に対応して
いるので、切れたら同じコマンドを打ち直せばよい。何度も切れて進まないなら、
それはログインを避けることのコストである。トークンは使わない方針なので、
時間をずらして打ち直すか、下のHF以外の経路に替える。

**HFを一度も使わない道もある。**

|経路|アカウント|取れるもの|注意|
|---|---|---|---|
|`ollama pull`|**不要**|`bge-m3`、`gpt-oss`、`qwen3-vl` など|local-llm-ragの現構成そのもの。[8節](#8-local-llm-ragのragとつなぐ)の道(a)（RAGはOllama、vLLMはエージェント）を採るなら、**HFに一度も触れずに済む**。ただしNemotron系のNVFP4は置かれていない|
|NGC|環境による|Nemotron系などNVIDIA製モデル|公開アーティファクトはゲストモードで取れるとされるが、APIキーを求められる報告もある。社内でNGCを使っているなら試す価値がある|
|ModelScope|不要|多くの公開モデル|vLLMが `VLLM_USE_MODELSCOPE=True` で対応する。ただし**外部の宛先が1本増える**。送るのはモデルファイルの要求だけで社内資料は出ないが、`AGENTS.md` の「外部へ出る通信」の一覧に追記すべき経路である。中国のサービスである点も、社内規程次第では判断材料になる|
|別マシンで取って搬入|—|すべて|上のオフライン手順。匿名で取れるモデルなら、そのマシンでもログインは要らない|

> **確認していないこと**: この文書を書いた環境から huggingface.co へ到達できて
> いないため、**個々のモデルのゲート有無は確認していない**。一般論として、
> `openai/gpt-oss-*`（Apache 2.0）、`BAAI/bge-m3`（MIT）、Qwen系はゲート無しで
> 配られており、NVIDIAのNemotron系もNVIDIA Open Model Licenseで多くはゲート無し
> である。ただしNVIDIAのリポジトリにも同意や申請を要求するものが実在する。
> **上の1行で各自確認すること。**

## 6. コーディングエージェント向けの起動

playbookが載せている「Agent Ready」のレシピをそのまま引く。ツール呼び出しの
パーサーと推論（reasoning）パーサーの指定が入っているのが要点で、これが無いと
エージェントからはツールが使えない。

```bash
docker run -d --name vllm-code --restart unless-stopped \
  --gpus all --ipc=host \
  -p 127.0.0.1:11434:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  nvcr.io/nvidia/vllm:26.10-py3 \
  vllm serve nvidia/Qwen3.6-35B-A3B-NVFP4 \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --kv-cache-dtype fp8 \
    --attention-backend flashinfer \
    --moe-backend marlin \
    --gpu-memory-utilization 0.4 \
    --max-model-len 262144 \
    --max-num-seqs 4 \
    --max-num-batched-tokens 8192 \
    --enable-chunked-prefill \
    --async-scheduling \
    --enable-prefix-caching \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}' \
    --load-format fastsafetensors \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_xml \
    --enable-auto-tool-choice
```

playbookのレシピは `vllm/vllm-openai:latest` を使っているが、ここでは2節と同じ
NGCのタグ `26.10-py3` に固定し、`-e HF_TOKEN` も渡さない。NGCイメージは
エントリーポイントが `vllm serve` ではないので、`vllm serve` から書く。
**フラグの中にこの版のvLLMが知らないものがあれば起動時にエラーになる**ので、
そのときは `docker run --rm nvcr.io/nvidia/vllm:26.10-py3 vllm serve --help` で確かめ、
該当するフラグ（`--moe-backend` や `--speculative-config` の `moe_backend` など）を外す。

ポートは、Codex などクライアントPCから使うので外部向けの 11434 番にする
（LANへの公開は[4節](#4-社内ネットワークへの公開と認証)）。2節の `vllm-chat` と同時に
置くなら 11434 番は1本にしか使えないので、外に出すほうを 11434 番、もう一方を
`-p 127.0.0.1:8001:8000` にする（[ポートの決まり](#ポートの決まり)）。

エージェント用途で効いているフラグ:

- `--enable-auto-tool-choice` / `--tool-call-parser` — ツール呼び出しをOpenAI互換の
  `tool_calls` として返させる。**エージェントを使うなら必須**
- `--reasoning-parser` — 思考部分を `reasoning_content` に分離する。これが無いと
  思考がそのまま本文に混ざる
- `--max-model-len 262144` — コードベースを読ませるための長い文脈
- `--max-num-seqs 4` — 1人がエージェントを回す前提。同時利用者が増えるなら上げる

### クライアント側の設定

vLLMが出すのは **OpenAI互換API**（`/v1/chat/completions`、`/v1/completions`、
`/v1/embeddings`、`/v1/responses` ほか）である。これに合わせられるツールなら繋がる。

- **Codex CLI** — local-llm-ragが `infra/codex-colab/config.toml.template` で
  組み立てているのと同じ形で、`base_url` をvLLMに向ける。

  ```toml
  model = "nvidia/Qwen3.6-35B-A3B-NVFP4"
  model_provider = "gb10-vllm"
  model_context_window = 262144

  [model_providers.gb10-vllm]
  name = "GB10 vLLM"
  base_url = "http://<GB10のアドレス>:11434/v1"
  wire_api = "chat"          # vLLMは /v1/responses も出すが、まずはchatで確かめる
  env_key = "VLLM_API_KEY"   # --api-key を付けた場合。Bearerで送られる
  ```

- **Cline / Continue / Roo Code など** — 「OpenAI Compatible」のプロバイダーを選び、
  Base URLに `http://<GB10のアドレス>:11434/v1`、モデル名にHFハンドルを入れる。
- **Claude Code** — Anthropic形式のAPIを話すので、**vLLMへ直接は向けられない**。
  使うならAnthropic形式↔OpenAI形式を変換するプロキシを別途挟むことになる。
  GB10のローカルモデルで完結させたいなら、OpenAI互換のエージェントを選ぶほうが素直である。

> **注意** local-llm-ragの `scripts/coding_agent.py`（`coding_agent/`）は、
> ColabのOllamaに繋ぐ専用の起動補助である。接続先にHTTPSを必須とし、
> `/api/tags` `/api/create` `/api/ps` といったOllama固有のAPIでコンテキスト長を
> 設定し、モデル名も `gpt-oss:20b` に固定している（`coding_agent/connection.py`）。
> **vLLMには向けられない。** vLLMを使うなら、上の `config.toml` を手で置くか、
> vLLM向けの経路をコードに足す必要がある。

## 7. 埋め込みとリランカーをvLLMで出す

vLLMは生成だけでなく、埋め込み（pooling）とリランクも出せる。

```bash
# 埋め込み（/v1/embeddings）
docker run -d --name vllm-embed --restart unless-stopped \
  --gpus all --ipc=host -p 127.0.0.1:8002:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ${VLLM_IMAGE} \
  vllm serve BAAI/bge-m3 \
    --host 0.0.0.0 --port 8000 \
    --runner pooling \
    --gpu-memory-utilization 0.05
```

```bash
curl http://localhost:8002/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "BAAI/bge-m3", "input": ["これは埋め込みの確認である"]}'
```

埋め込みは2本目のvLLMなので、ホストの `127.0.0.1:8002` にだけ出している（コンテナの中は
8000 番）。クライアントPCの RAG から使うなら、ファイアウォールは開けずに、4000番の LiteLLM に
`model_name` で並べて出す（[ポートの決まり](#ポートの決まり)）。

`--runner pooling` の指定名は版で変わっている（古い版は `--task embed`）。
通らなければ `vllm serve --help` で確認する。bge-m3のsparse/ColBERT側の重みまで
使いたい場合は `--hf-overrides '{"architectures": ["BgeM3EmbeddingModel"]}'` が要るが、
local-llm-ragは**dense（1024次元）しか使っていない**ので、その指定は不要である。

リランカーは `BAAI/bge-reranker-v2-m3` を同じくpoolingで起動すると
`/v1/rerank`（Jina/Cohere互換）と `/v1/score` が使える。ただしlocal-llm-ragの
リランカーは `onnxruntime` のINT8でCPU実行しており、1問あたり約1.3秒
（README記載の実測値）。GB10へ移せば速くはなるが、**移行の必須項目ではない**。
生成の移行が終わってから考えればよい。

## 8. local-llm-ragのRAGとつなぐ

**ここが一番の落とし穴である。`OLLAMA_HOST` にvLLMのURLを入れても動かない。**

local-llm-ragはOllamaの**ネイティブAPI**を直接叩いている。

|呼び出し元|叩いているURL|
|---|---|
|`ingest/chat.py`（`ask_json` / `ask_text` / `stream_chat`）|`{OLLAMA_HOST}/api/chat`|
|`ingest/embedder.py`（`embed_texts` / `embed_query`）|`{OLLAMA_HOST}/api/embed`|

vLLMが出すのは `/v1/...` のOpenAI互換APIだけで、`/api/chat` も `/api/embed` も無い。
`.env` の `OLLAMA_HOST` をvLLMに向ければ、404が返るだけである。
（`coding_agent/` だけは `base_url` に `/v1` を付けており、これはOllamaの
OpenAI互換エンドポイントを使っている。RAG本体とは別経路である。）

取りうる道は3つある。

**(a) GB10にOllamaも入れ、RAGはOllama・エージェントはvLLM（改修ゼロ・推奨の第一歩）**

DGX Spark playbookには `ollama` の項目もあり、GB10でOllamaは動く。RAG側は
`.env` の `OLLAMA_HOST` をGB10のOllamaに向けるだけで済む（[README「ColabのL4 GPUに
接続する」](https://github.com/Hide369/local-llm-rag/blob/master/README.md#colabのl4-gpuに接続する)と同じ仕組みで、向き先が
ngrokからLANのアドレスに変わるだけである）。vLLMはコーディングエージェント専用に
使う。**まずこれで動かし、RAGの生成をvLLMへ寄せるかは実測してから決める**のが安全である。
Ollamaとの併存では、両者が同じ128GBを食い合う点に注意する。**ポートもぶつかる。**
LANに出せるのは 11434 番と 4000 番だけなので、この道では 11434 番を Ollama に充て
（`OLLAMA_HOST=http://<GB10のIP>:11434`）、vLLM は `-p 127.0.0.1:8001:8000` でループバックに
置いて 4000 番の LiteLLM 経由でエージェントに出す（[ポートの決まり](#ポートの決まり)）。

**(b) OpenAI互換の経路をコードに足す（本命）**

変更が要るのは実質4か所（`ask_json` / `ask_text` / `stream_chat` / `embed_texts`）で、
`OLLAMA_HOST` とは別に `OPENAI_BASE_URL` のような設定を読み、`/v1/chat/completions` と
`/v1/embeddings` を叩く実装を選べるようにする。Ollama経路を消さずに足すこと
（ノートPC単体での利用と、Colab接続が生きているため）。

**(c) 変換プロキシを挟む**

Ollamaネイティブ形式をOpenAI形式へ翻訳する層を自前で立てる。依存と障害点が増える
だけで、(b) より優れる点が無い。採らない。

### 埋め込みを移すときの必須手順

**埋め込みの経路を変えたら、ベクトルDBを作り直すこと。** 同じ `bge-m3` でも、
Ollama（GGUF量子化）とvLLM（safetensors）では出てくるベクトルが完全一致しない。
DBに入っているベクトルと問い合わせのベクトルは同じ経路で作られている必要がある。

```powershell
# 埋め込みの接続先を変えたら、差分ではなく全件を取り込み直す
.\myvenv313\Scripts\python.exe -m scripts.ingest_source --force
```

生成（`/api/chat`）だけを差し替える場合は、DBの作り直しは要らない。
**移行は「生成だけ先に」「埋め込みは後で、再取り込みとセットで」の順が安全である。**

### 外部通信の観点

`AGENTS.md` に、この構成が外へ出す通信の一覧がある。GB10のvLLMを社内に置いた場合、
`OLLAMA_HOST` 相当の宛先が**社内のLANアドレスになる**ため、質問文・会話履歴・
検索でヒットしたチャンク本文・取り込み時の文書本文がマシンの外へ出る範囲は
**社内ネットワークの中に収まる**。ColabのngrokへL4を借りていた構成より、
この点は明確に改善する。

ただし初回のモデル取得は huggingface.co への通信である。オフライン運用にするなら、
`~/.cache/huggingface` を作ったうえでネットワークを切る、という順序で用意する。
リランカーモデルのHEADリクエスト（`ingest/reranker.py`）についても同じ扱いになる。

## 9. トラブルシュート

|症状|原因|対処|
|---|---|---|
|起動時にCUDAカーネル初期化で落ちる|PyTorch/Tritonが `sm_120` 止まりのビルド|NGCの `nvcr.io/nvidia/vllm` か、GB10向けにビルドされたイメージを使う。素のpip導入を避ける|
|メモリはあるはずなのにOOM|UMAのページキャッシュが居座っている|`sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` でバッファを流す（playbook記載）|
|`CUDA out of memory`|`--max-model-len` / `--max-num-seqs` が大きい|どちらも下げる。複数プロセス常駐なら `--gpu-memory-utilization` の合計を見直す。切り分けの手順は [vllm-gb10-memory.md](vllm-gb10-memory.md)|
|`Cannot access gated repo` / `401`|ゲート付きのモデルを指定している|トークンは使わない方針なので、そのモデルは使わない。[入手経路とオフライン運用](#入手経路とオフライン運用)の1行でゲートの無い公開モデルを選び直す|
|エージェントがツールを呼ばない|`--enable-auto-tool-choice` と `--tool-call-parser` が無い|[6節](#6-コーディングエージェント向けの起動)のレシピに合わせる。パーサー名はモデルごとに違う|
|`address already in use`（11434）|同じGB10でOllamaが11434番を使っている|Ollamaを止めるか、[ポートの決まり](#ポートの決まり)のとおりvLLMをループバックへ置き、4000番のLiteLLM経由にする|
|クライアントPCから11434番に届かない|`-p 127.0.0.1:11434:8000` のまま、または `DOCKER-USER` の規則で塞いでいる|[4節](#4-社内ネットワークへの公開と認証)。`docker ps` で `0.0.0.0:11434->8000/tcp` になっているかを見る|
|コンテナを作り直すたびに巨大なDL|HFキャッシュを未マウント|`-v ~/.cache/huggingface:/root/.cache/huggingface` を必ず付ける|
|初回起動が異様に遅い|モデルDLとカーネルのコンパイル|2回目以降は速くなる。`--load-format fastsafetensors` でロード自体も短縮できる|
|`.env` の `OLLAMA_HOST` を向けたのにRAGが404|vLLMに `/api/chat` は無い|[8節](#8-local-llm-ragのragとつなぐ)。設定では解決しない。経路の追加が要る|
|GB10のデスクトップで日本語が打てない|IMEが未導入（vLLMとは無関係）|[docs/gb10-japanese-input.md](https://github.com/Hide369/local-llm-rag/blob/master/docs/gb10-japanese-input.md)|

2台以上のGB10をQSFPで繋いで `--tensor-parallel-size 2` 以上で動かす手順
（Rayクラスタ）もplaybookにある。1台に載らないモデルを動かす必要が出てから読めばよい。
4台をスイッチで繋ぐ場合は [GB10を4台スイッチで繋ぎ、vLLMで1つのモデルを分散して動かす](../vllm_gb10_cluster.md)にまとめた。

## 出典

- [NVIDIA DGX Spark playbook — vLLM for Inference](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/vllm/README.md)（コンテナのタグ、起動フラグ、Agent Readyレシピ、`drop_caches`、Rayクラスタ手順の出どころ。2026-06-12 更新）
- [NVIDIA NGC — vLLM コンテナのタグ一覧](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm/tags)（この文書は `26.10-py3` を使う）
- Batch_Exec 内: [vllm_migration.md](../../pr33-38/vllm_migration.md)（1台で Ollama を止めて vLLM に切り替える。同じイメージタグ、同じ 11434→8000 のポート）
- [vLLM — OpenAI互換サーバー](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/)
- [vLLM — Pooling Models（埋め込み・リランク）](https://docs.vllm.ai/en/latest/models/pooling_models/)
- [vLLM issue #36821 — sm_121 / aarch64 でのビルド問題](https://github.com/vllm-project/vllm/issues/36821)
- [vLLM issue #31128 — Blackwell SM121(DGX Spark) 対応](https://github.com/vllm-project/vllm/issues/31128)
- [timothystewart6/vllm-gb10 — GB10（sm_121a）向けにビルドされた非公式イメージ](https://github.com/timothystewart6/vllm-gb10)
- [HuggingFace — 環境変数（`HF_HOME` / `HF_HUB_OFFLINE`）](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables)（オフライン運用の出どころ）
- [jschmied/dgx-spark-setup-guide](https://github.com/jschmied/dgx-spark-setup-guide) / [timothystewart6/ubuntu-gb10](https://github.com/timothystewart6/ubuntu-gb10)（DGX OSの版とSSHでの初期セットアップ）
- local-llm-rag内: [docs/gb10-japanese-input.md](https://github.com/Hide369/local-llm-rag/blob/master/docs/gb10-japanese-input.md)（GB10のデスクトップの日本語入力）、
  [docs/server-deployment.md](https://github.com/Hide369/local-llm-rag/blob/master/docs/server-deployment.md)（公開構成の考え方）、
  [README「ColabのL4 GPUに接続する」](https://github.com/Hide369/local-llm-rag/blob/master/README.md#colabのl4-gpuに接続する)（`OLLAMA_HOST` の差し替え）、
  `ingest/chat.py` / `ingest/embedder.py`（Ollamaネイティブapiの呼び出し箇所）
