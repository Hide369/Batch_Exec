r"""言語に依存しないソース処理（字句の分割・コメント以外の差分検出・抽出結果の型）。

TODO(技術的負債): 字句の分割は C 系の言語に共通する規則（// と /* */ のコメント、
" ' ` の文字列、C# の @"..."）だけを見る簡易版。C# の生文字列リテラル（" を3つ以上重ねた囲み）、
TypeScript のテンプレート文字列中の \` や正規表現リテラル（/\/*/ など）は正しく分割できないことがある。
"""

import hashlib
import re
from collections.abc import Iterator
from dataclasses import dataclass, field

# 識別子・数値はひとかたまり、記号は1文字ずつ。"a+b" と "a + b" を同じトークン列にする
_CODE_TOKEN_RE = re.compile(r"\w+|\n|[^\w\s]")


@dataclass(frozen=True)
class SourceFacts:
    """静的解析で抽出した、LLM に渡す「事実」。"""

    language: str  # プロンプトに書く言語名（例: Go）
    package: str | None  # package / namespace（ない言語は None）
    imports: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    types: list[str] = field(default_factory=list)
    # 設計書に必ず記載されるべき公開識別子。何を公開とみなすかは言語ごとに決まる
    exported: list[str] = field(default_factory=list)
    package_label: str = "package"

    def to_prompt(self) -> str:
        def fmt(items: list[str]) -> str:
            return ", ".join(items) if items else "(なし)"

        lines = [f"- {self.package_label}: {self.package}"] if self.package is not None else []
        lines += [
            f"- imports: {fmt(self.imports)}",
            f"- types: {fmt(self.types)}",
            f"- functions/methods: {fmt(self.functions)}",
        ]
        return "\n".join(lines)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def scan(source: str) -> Iterator[tuple[str, str, int]]:
    """ソースを ("comment" | "string" | "code", 文字列, 開始行) に分ける。

    文字列リテラルを区別するのは、"http://..." の "//" をコメントと誤認しないため。
    """
    i, line, n = 0, 1, len(source)
    while i < n:
        c = source[i]
        if source.startswith("//", i):
            end = source.find("\n", i)
            kind, j = "comment", n if end == -1 else end
        elif source.startswith("/*", i):
            end = source.find("*/", i + 2)
            kind, j = "comment", n if end == -1 else end + 2
        elif source.startswith(('@"', '@$"'), i):
            # C# の逐語的文字列 @"C:\temp\"。\ はエスケープにならず、"" が " を表す
            j = source.index('"', i) + 1
            while j < n:
                if source[j] == '"':
                    if source.startswith('""', j):
                        j += 2
                        continue
                    break
                j += 1
            kind, j = "string", min(j + 1, n)
        elif c == "`":
            # Go の生文字列（\ はエスケープにならない）と TypeScript のテンプレート文字列
            end = source.find("`", i + 1)
            kind, j = "string", n if end == -1 else end + 1
        elif c in "\"'":
            j = i + 1
            while j < n and source[j] not in (c, "\n"):
                j += 2 if source[j] == "\\" else 1
            kind, j = "string", min(j + 1, n)
        else:
            j = i + 1
            while j < n and source[j] not in "/`\"'" and not source.startswith(('@"', '@$"'), j):
                j += 1
            kind = "code"
        text = source[i:j]
        yield kind, text, line
        line += text.count("\n")
        i = j


def strip_comments(source: str) -> str:
    # コメント内の "func Foo(" などを誤検出しないよう除去する（行番号は保つ）
    return "".join("\n" * text.count("\n") if kind == "comment" else text for kind, text, _ in scan(source))


def strip_comments_and_strings(source: str) -> str:
    # 文字列内の "class Foo" なども誤検出しないよう、文字列は中身を空にした "" に置き換える（行番号は保つ）
    def keep(kind: str, text: str) -> str:
        if kind == "code":
            return text
        return ('""' if kind == "string" else "") + "\n" * text.count("\n")

    return "".join(keep(kind, text) for kind, text, _ in scan(source))


def _code_tokens(source: str) -> list[tuple[str, int]]:
    """コメントと空白の違いを無視して比較するための (トークン, 行) の列。

    改行で文末を補う言語（Go・TypeScript）があるため改行だけは残し、連続する改行は1つにまとめる。
    """
    tokens: list[tuple[str, int]] = []

    def newline(line: int) -> None:
        if tokens and tokens[-1][0] != "\n":
            tokens.append(("\n", line))

    for kind, text, line in scan(source):
        if kind == "comment":
            if "\n" in text:  # 改行を含むブロックコメントは改行として扱われる
                newline(line)
        elif kind == "string":
            tokens.append((text, line))
        else:
            for m in _CODE_TOKEN_RE.finditer(text):
                tok_line = line + text.count("\n", 0, m.start())
                if m.group() == "\n":
                    newline(tok_line)
                else:
                    tokens.append((m.group(), tok_line))
    if tokens and tokens[-1][0] == "\n":
        tokens.pop()
    return tokens


@dataclass(frozen=True)
class CodeDifference:
    """コメントと空白以外の最初の違い。再生成の依頼とログに、どこがどう違うかを具体的に示すために使う。"""

    line: int  # 元のソースでの行番号
    modified_line: int | None  # 出力での行番号（出力が途中で終わっていれば None）
    expected: str | None  # 元のソースのトークン（出力の末尾に余分なコードがあれば None）
    actual: str | None  # 出力のトークン（出力が途中で終わっていれば None）
    original_text: str  # 元のソースのその行
    modified_text: str  # 出力のその行

    def describe(self) -> str:
        def tok(t: str) -> str:
            return "改行" if t == "\n" else f"`{t}`"

        def quote(text: str) -> str:
            text = text.strip()
            return f"`{text[:157] + '…' if len(text) > 160 else text}`" if text else "（空行）"

        if self.actual is None:
            return (f"出力が元のファイルの {self.line} 行目付近で終わっています（元: {quote(self.original_text)} 以降がありません）。"
                    "ファイルを省略せず、最後まで出力してください")
        if self.expected is None:
            return (f"元のファイルにないコードが末尾に追加されています（出力の {self.modified_line} 行目: "
                    f"{quote(self.modified_text)}）")
        return (f"元のファイルの {self.line} 行目付近で {tok(self.expected)} が {tok(self.actual)} に変わっています。"
                f"元: {quote(self.original_text)} / 出力（{self.modified_line} 行目）: {quote(self.modified_text)}")


def _line_text(source: str, line: int) -> str:
    lines = source.splitlines()
    return lines[line - 1] if 0 < line <= len(lines) else ""


def find_code_difference(original: str, modified: str) -> CodeDifference | None:
    """コメントと空白以外に違いがあれば、最初の違いを返す。"""
    before, after = _code_tokens(original), _code_tokens(modified)

    def diff(line: int, mod_line: int | None, expected: str | None, actual: str | None) -> CodeDifference:
        return CodeDifference(
            line, mod_line, expected, actual,
            _line_text(original, line), _line_text(modified, mod_line) if mod_line else "",
        )

    for (b, line), (a, mod_line) in zip(before, after):
        if a != b:
            return diff(line, mod_line, b, a)
    if len(before) > len(after):
        # 改行トークンは前の行に属するため、削られた最初のコードの行を示す
        tok, line = next(((t, ln) for t, ln in before[len(after):] if t != "\n"), before[-1])
        return diff(line, None, tok, None)
    if len(before) < len(after):
        tok, mod_line = next(((t, ln) for t, ln in after[len(before):] if t != "\n"), after[-1])
        return diff(before[-1][1] if before else 1, mod_line, None, tok)
    return None


def first_code_difference(original: str, modified: str) -> int | None:
    """コメントと空白以外に違いがあれば、元のソースでの最初の差分の行番号を返す。"""
    found = find_code_difference(original, modified)
    return found.line if found else None
