# GB10 のテキストエディタで日本語を入力できないときの対処

GB10（DGX OS。Ubuntu 24.04 ベースの GNOME デスクトップ）で、**端末などでは日本語を入力できるのに、GUI のテキストエディタ（Text Editor）だけ入力できない**ときの切り分けと対処。
バッチ（runner）・Web 画面・推論サーバー（vLLM）の設定は変えない。OS のデスクトップ側の設定だけで行う。

## なぜテキストエディタだけ入力できないのか

Ubuntu 24.04 の標準テキストエディタ（GNOME Text Editor、コマンド名 `gnome-text-editor`）は **GTK4** で作られている。端末（GNOME Terminal）などの多くのアプリは **GTK3** で作られている。
日本語入力（IME）は GTK のバージョンごとに別の接続部品を使うため、GTK3 側だけ繋がっていると「他のアプリでは入力できるが、テキストエディタだけ入力できない」状態になる。

```
キー入力 ──> IME（fcitx5 または IBus） ──┬─ GTK3 用の接続部品 ──> 端末など（入力できる）
                                          └─ GTK4 用の接続部品 ──> Text Editor（ここが欠けている）
```

よくある原因は次の2つ。

| 原因 | 起きる条件 |
|---|---|
| A. GTK4 用の接続部品が無い | IME に **fcitx5**（`fcitx5-mozc`）を使っていて、`fcitx5-frontend-gtk4` が入っていない |
| B. エディタが隔離されている | テキストエディタが **Snap / Flatpak** 版で、IME に繋がらない |

コマンドは GB10 の端末で実行する（SSH は使えないため、GB10 に直接ログインした端末を使う）。`sudo` が必要な手順がある。
途中でログアウト（または再起動）が要る。実行中のバッチがあれば終わってから行う（Web 画面・vLLM はサービスなので、ログアウトしても止まらない）。

## 1. 原因を切り分ける

```bash
im-config -m                                   # 2行目が使われている IME（fcitx5 / ibus）
ps -e | grep -E 'fcitx|ibus' | sort -u -k4     # 実際に動いている IME
echo "GTK_IM_MODULE=$GTK_IM_MODULE  XDG_SESSION_TYPE=$XDG_SESSION_TYPE"
dpkg -l | grep -E 'fcitx5-frontend|ibus-gtk'   # 入っている接続部品
snap list 2>/dev/null | grep -iE 'text-editor|gedit'      # Snap 版のエディタがあるか
flatpak list 2>/dev/null | grep -iE 'TextEditor|gedit'    # Flatpak 版のエディタがあるか
```

| 結果 | 原因 | 対処 |
|---|---|---|
| IME が `fcitx5` で、`fcitx5-frontend-gtk4` が一覧に無い | A | 2. |
| `snap list` / `flatpak list` にテキストエディタが出る | B | 3. |
| IME が `ibus` で、Snap / Flatpak 版でもない | GTK4 用の部品・設定の不足 | 4. |
| どれにも当てはまらない、または急いで使いたい | — | 5.（gedit を使う） |

## 2. 原因 A：fcitx5 の GTK4 用の部品を入れる

```bash
sudo apt update
sudo apt install -y fcitx5-frontend-gtk4 fcitx5-frontend-gtk3
```

入れたら **ログアウトしてログインし直す**（確実にするなら再起動）。テキストエディタを開き、日本語を入力できるか確かめる（6.）。

## 3. 原因 B：Snap / Flatpak 版のエディタを通常版（deb）に替える

```bash
sudo apt install -y gnome-text-editor
# Snap 版を消す場合（1. で snap list に出た名前を使う）
sudo snap remove gnome-text-editor
# Flatpak 版を消す場合
flatpak uninstall org.gnome.TextEditor
```

アプリ一覧に同じ名前のエディタが2つ出ている場合は、deb 版（`/usr/bin/gnome-text-editor`）で開く。端末から `/usr/bin/gnome-text-editor <ファイル>` と起動すれば確実に deb 版になる。

## 4. IBus を使っている場合

```bash
sudo apt update
sudo apt install -y ibus-gtk4 ibus-gtk3
```

入れたらログアウトしてログインし直す。
それでも入力できない場合、`GTK_IM_MODULE` が `xim` など `ibus` 以外になっていないか確かめる（1. の出力）。`~/.profile`・`~/.bashrc`・`/etc/environment` で設定していれば、その行を消すか `GTK_IM_MODULE=ibus` に直してログインし直す。

## 5. すぐに使いたい場合：gedit を使う

gedit は GTK3 で作られているため、端末と同じ仕組みで日本語を入力できる。原因を直すまでの代わりとしても、そのまま使い続けてもよい。

```bash
sudo apt install -y gedit
gedit <ファイル>
```

ファイルマネージャーでテキストファイルを右クリック → **Open With（別のアプリで開く）** → gedit を選び、「常にこのアプリで開く」にすれば、ダブルクリックで gedit が開く。

## 6. 入力できるか確かめる

1. テキストエディタで新しいファイルを開き、IME を日本語に切り替えて「にほんご」と打ち、Space で「日本語」に変換できること
2. 保存して、端末で中身を確かめる

```bash
cat <保存したファイル>     # 入力した日本語がそのまま表示されること
file <保存したファイル>    # "UTF-8 text" と出ること
```

## 7. 元に戻す

入れたパッケージは、ほかのアプリの動きを変えないため、残しておいてよい。消す場合は次のとおり（2.・4.・5. で入れたものだけ）。

```bash
sudo apt remove -y fcitx5-frontend-gtk4      # 2. で入れた場合
sudo apt remove -y ibus-gtk4                 # 4. で入れた場合
sudo apt remove -y gedit                     # 5. で入れた場合
```

4. で `GTK_IM_MODULE` の行を直した場合は、元の行に戻してログインし直す。
