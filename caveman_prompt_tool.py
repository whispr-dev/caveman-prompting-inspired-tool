#!/usr/bin/env python3
"""
caveman_prompt_tool.py

Prompt compressor inspired by JuliusBrussee/caveman and caveman-compress.

What this does:
- Compresses natural-language prompt and memory files.
- Preserves YAML frontmatter, markdown headings, fenced code, inline code,
  URLs, links, env vars, file paths, and many technical spans as safely as possible.
- Can emit "caveman overlay" instruction blocks for Claude and other agents.
- Supports a conservative local compressor and an optional Anthropic/Claude
  backend for stronger compression closer to the repo's intent.

Safe default:
- Dry run unless you pass --write or --out-dir.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

TEXT_EXTENSIONS = {".md", ".markdown", ".txt", ".rst"}
SKIP_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".yaml", ".yml", ".toml",
    ".env", ".lock", ".css", ".scss", ".sass", ".less", ".html", ".xml",
    ".sql", ".sh", ".bash", ".zsh", ".ps1", ".psm1", ".bat", ".cmd", ".c",
    ".cpp", ".cc", ".cxx", ".h", ".hpp", ".java", ".go", ".rs", ".php",
    ".rb", ".pl", ".lua", ".swift", ".kt", ".scala", ".ini", ".cfg"
}

FILLER_WORDS = {
    "just", "really", "basically", "actually", "simply", "essentially",
    "generally", "quite", "very", "fairly", "rather"
}

LONG_PHRASE_REPLACEMENTS = [
    (r"\bit is important to note that\b", "note:"),
    (r"\bplease note that\b", "note:"),
    (r"\bit should be noted that\b", "note:"),
    (r"\bthe reason (?:why )?is because\b", "because"),
    (r"\bdue to the fact that\b", "because"),
    (r"\bin order to\b", "to"),
    (r"\bmake sure to\b", "ensure"),
    (r"\byou should make sure to\b", "ensure"),
    (r"\byou should\b", ""),
    (r"\byou can\b", ""),
    (r"\byou may\b", ""),
    (r"\byou might\b", ""),
    (r"\bit would be good to\b", ""),
    (r"\bit might be worth\b", ""),
    (r"\byou could consider\b", ""),
    (r"\bi would recommend\b", ""),
    (r"\bi recommend\b", ""),
    (r"\bi'd recommend\b", ""),
    (r"\bi'd suggest\b", ""),
    (r"\bi suggest\b", ""),
    (r"\bi'd be happy to\b", ""),
    (r"\bhappy to\b", ""),
    (r"\bof course\b", ""),
    (r"\bcertainly\b", ""),
    (r"\bsure\b", ""),
    (r"\bfor the purpose of\b", "for"),
    (r"\bis able to\b", "can"),
    (r"\bis responsible for\b", "manages"),
    (r"\bis used to\b", "used to"),
    (r"\bhelps to\b", "helps"),
    (r"\bhas the ability to\b", "can"),
    (r"\bdo not forget to\b", "remember to"),
    (r"\bkeep in mind that\b", ""),
    (r"\bin the event that\b", "if"),
    (r"\bat this point in time\b", "now"),
    (r"\bin close proximity to\b", "near"),
    (r"\ba large number of\b", "many"),
    (r"\ba significant number of\b", "many"),
]

WORD_REPLACEMENTS = {
    "utilize": "use",
    "utilized": "used",
    "utilizing": "using",
    "approximately": "about",
    "modification": "change",
    "modifications": "changes",
    "additional": "more",
    "assistance": "help",
    "purchase": "buy",
    "obtain": "get",
    "regarding": "about",
    "commence": "start",
    "terminate": "end",
    "facilitate": "help",
    "demonstrate": "show",
    "indicates": "shows",
    "indicate": "show",
    "numerous": "many",
    "therefore": "so",
    "however": "but",
    "furthermore": "also",
    "additionally": "also",
    "perform": "do",
    "requires": "needs",
    "require": "need",
    "attempt": "try",
    "validate": "check",
    "validation": "check",
    "configuration": "config",
    "authenticate": "auth",
    "authentication": "auth",
    "application": "app",
    "applications": "apps",
    "database": "DB",
    "connection": "conn",
    "connections": "conns",
    "environment": "env",
    "parameters": "params",
    "parameter": "param",
}

ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
MULTISPACE_RE = re.compile(r"[ \t]{2,}")
SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")

INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\([^)]+\)")
URL_RE = re.compile(r"https?://[^\s)>]+")
ENV_VAR_RE = re.compile(r"(?<!\w)(?:\$[A-Z_][A-Z0-9_]*|[A-Z_][A-Z0-9_]*=[^\s]+)")
UNIX_PATH_RE = re.compile(r"(?<!\w)(?:\./|\../|/)[A-Za-z0-9._/\-]+")
WINDOWS_PATH_RE = re.compile(r"(?<!\w)[A-Za-z]:\\(?:[^\\\s]+\\)*[^\\\s]*")
VERSION_RE = re.compile(r"\bv?\d+(?:\.\d+){1,3}\b")
NUMERIC_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")

PROTECTED_PATTERNS = [
    INLINE_CODE_RE,
    MARKDOWN_LINK_RE,
    URL_RE,
    ENV_VAR_RE,
    WINDOWS_PATH_RE,
    UNIX_PATH_RE,
    VERSION_RE,
    NUMERIC_RE,
]

BULLET_RE = re.compile(r"^(\s*(?:[-*+]|\d+[.)])\s+)(.*)$")
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")
FENCE_RE = re.compile(r"^(\s*)(`{3,}|~{3,})(.*)$")
TABLE_RE = re.compile(r"^\s*\|.*\|\s*$")
BLOCKQUOTE_RE = re.compile(r"^\s*>\s?")
HTML_TAG_RE = re.compile(r"^\s*<[^>]+>\s*$")


@dataclasses.dataclass
class Segment:
    kind: str
    text: str


@dataclasses.dataclass
class Report:
    path: str
    wrote: bool
    backend: str
    backup: Optional[str]
    output: Optional[str]
    original_chars: int
    compressed_chars: int
    original_words: int
    compressed_words: int
    char_savings_pct: float
    word_savings_pct: float
    validation_ok: bool
    skipped_reason: Optional[str] = None


class CompressionError(RuntimeError):
    pass


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def insert_before_suffix(path: Path, insert: str) -> Path:
    if path.suffix:
        return path.with_name(f"{path.stem}{insert}{path.suffix}")
    return path.with_name(path.name + insert)


def should_process(path: Path) -> Tuple[bool, Optional[str]]:
    if not path.exists():
        return False, "file does not exist"
    if path.is_dir():
        return False, "path is a directory"
    if ".original" in path.name:
        return False, "backup/original file skipped"
    suffix = path.suffix.lower()
    if suffix in SKIP_EXTENSIONS:
        return False, f"code/config extension skipped: {suffix}"
    if suffix and suffix not in TEXT_EXTENSIONS:
        return False, f"unsupported extension skipped: {suffix}"
    return True, None


def protect_spans(text: str) -> Tuple[str, Dict[str, str]]:
    protected: Dict[str, str] = {}
    counter = 0

    def make_repl(match: re.Match[str]) -> str:
        nonlocal counter
        key = f"__CAVEMAN_PROTECTED_{counter}__"
        counter += 1
        protected[key] = match.group(0)
        return key

    output = text
    for pattern in PROTECTED_PATTERNS:
        output = pattern.sub(make_repl, output)
    return output, protected


def restore_spans(text: str, protected: Dict[str, str]) -> str:
    output = text
    for key, value in protected.items():
        output = output.replace(key, value)
    return output


def normalize_spacing(text: str) -> str:
    s = MULTISPACE_RE.sub(" ", text)
    s = SPACE_BEFORE_PUNCT_RE.sub(r"\1", s)
    s = re.sub(r"\s*([=+\-→])\s*", r" \1 ", s)
    s = MULTISPACE_RE.sub(" ", s)
    return s.strip()


def compress_sentence_local(text: str, mode: str) -> str:
    if not text.strip():
        return text

    original = text
    protected_text, protected = protect_spans(text)
    s = protected_text

    for pattern, repl in LONG_PHRASE_REPLACEMENTS:
        s = re.sub(pattern, repl, s, flags=re.IGNORECASE)

    def replace_word(match: re.Match[str]) -> str:
        word = match.group(0)
        replacement = WORD_REPLACEMENTS.get(word.lower())
        if replacement is None:
            return word
        if word.isupper():
            return replacement.upper()
        if word[:1].isupper():
            return replacement[:1].upper() + replacement[1:]
        return replacement

    s = re.sub(r"\b[A-Za-z][A-Za-z\-]+\b", replace_word, s)

    if mode in {"full", "ultra"}:
        s = ARTICLES_RE.sub("", s)
        filler_pattern = r"\b(" + "|".join(re.escape(w) for w in sorted(FILLER_WORDS)) + r")\b"
        s = re.sub(filler_pattern, "", s, flags=re.IGNORECASE)

    s = re.sub(r"^\s*(?:please|kindly)\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(?:it|this|that) is important because\b", "because", s, flags=re.IGNORECASE)
    s = re.sub(r"\bit helps\b", "helps", s, flags=re.IGNORECASE)

    if mode in {"full", "ultra"}:
        s = re.sub(r"\b([A-Za-z][A-Za-z0-9_+\-/]*) is ([A-Za-z])", r"\1 \2", s)
        s = re.sub(r"\bthere is\b", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\bit is\b", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\bthis is\b", "this", s, flags=re.IGNORECASE)
        s = re.sub(r"\bthat is\b", "that", s, flags=re.IGNORECASE)

    if mode == "ultra":
        ultra_pairs = [
            (r"\bcauses\b", "→"),
            (r"\bleads to\b", "→"),
            (r"\bresults in\b", "→"),
            (r"\bbecause\b", "→"),
            (r"\bthen\b", "→"),
            (r"\band\b", "+"),
            (r"\bwith\b", "w/"),
            (r"\bwithout\b", "w/o"),
        ]
        for pattern, repl in ultra_pairs:
            s = re.sub(pattern, repl, s, flags=re.IGNORECASE)

    s = normalize_spacing(s)
    s = restore_spans(s, protected)
    if not s:
        return original.strip()
    return s


def compress_paragraph_local(text: str, mode: str) -> str:
    stripped = text.strip("\n")
    if not stripped.strip():
        return text

    lines = stripped.splitlines()
    compressed_lines: List[str] = []

    for line in lines:
        if not line.strip():
            compressed_lines.append("")
            continue

        bullet_match = BULLET_RE.match(line)
        if bullet_match:
            prefix, body = bullet_match.groups()
            compressed_lines.append(prefix + compress_sentence_local(body, mode))
            continue

        quote_match = BLOCKQUOTE_RE.match(line)
        if quote_match:
            marker = quote_match.group(0)
            body = line[len(marker):]
            compressed_lines.append(marker + compress_sentence_local(body, mode))
            continue

        compressed_lines.append(compress_sentence_local(line, mode))

    result = "\n".join(compressed_lines)
    if text.endswith("\n"):
        result += "\n"
    return result


def segment_text(text: str) -> List[Segment]:
    lines = text.splitlines(keepends=True)
    segments: List[Segment] = []

    i = 0
    n = len(lines)

    if n >= 1 and lines[0].strip() == "---":
        j = 1
        while j < n:
            if lines[j].strip() == "---":
                j += 1
                break
            j += 1
        segments.append(Segment("preserve", "".join(lines[:j])))
        i = j

    buffer: List[str] = []
    in_fence = False
    fence_marker = ""

    def flush_buffer() -> None:
        nonlocal buffer
        if buffer:
            segments.append(Segment("prose", "".join(buffer)))
            buffer = []

    while i < n:
        line = lines[i]

        if in_fence:
            buffer.append(line)
            if re.match(rf"^\s*{re.escape(fence_marker)}\s*$", line):
                segments.append(Segment("preserve", "".join(buffer)))
                buffer = []
                in_fence = False
                fence_marker = ""
            i += 1
            continue

        fence_match = FENCE_RE.match(line)
        if fence_match:
            flush_buffer()
            in_fence = True
            fence_marker = fence_match.group(2)
            buffer = [line]
            i += 1
            continue

        if HEADING_RE.match(line) or TABLE_RE.match(line) or HTML_TAG_RE.match(line):
            flush_buffer()
            segments.append(Segment("preserve", line))
            i += 1
            continue

        if line.startswith("    ") or line.startswith("\t"):
            flush_buffer()
            block = [line]
            i += 1
            while i < len(lines) and (lines[i].startswith("    ") or lines[i].startswith("\t")):
                block.append(lines[i])
                i += 1
            segments.append(Segment("preserve", "".join(block)))
            continue

        if not line.strip():
            flush_buffer()
            segments.append(Segment("preserve", line))
            i += 1
            continue

        buffer.append(line)
        i += 1

    flush_buffer()
    return segments


def extract_exact_headings(text: str) -> List[str]:
    return [m.group(0).rstrip("\n") for m in re.finditer(r"(?m)^\s{0,3}#{1,6}\s+.*$", text)]


def extract_fenced_blocks(text: str) -> List[str]:
    blocks: List[str] = []
    lines = text.splitlines(keepends=True)
    in_fence = False
    fence_marker = ""
    buffer: List[str] = []
    for line in lines:
        if not in_fence:
            m = FENCE_RE.match(line)
            if m:
                in_fence = True
                fence_marker = m.group(2)
                buffer = [line]
        else:
            buffer.append(line)
            if re.match(rf"^\s*{re.escape(fence_marker)}\s*$", line):
                blocks.append("".join(buffer))
                in_fence = False
                fence_marker = ""
                buffer = []
    return blocks


def extract_inline_code(text: str) -> List[str]:
    return INLINE_CODE_RE.findall(text)


def extract_urls(text: str) -> List[str]:
    urls = URL_RE.findall(text)
    urls.extend(m.group(0) for m in MARKDOWN_LINK_RE.finditer(text))
    return urls


def validate_preservation(original: str, compressed: str) -> Tuple[bool, List[str]]:
    errors: List[str] = []

    if extract_exact_headings(original) != extract_exact_headings(compressed):
        errors.append("headings changed")
    if extract_fenced_blocks(original) != extract_fenced_blocks(compressed):
        errors.append("fenced code blocks changed")
    if extract_inline_code(original) != extract_inline_code(compressed):
        errors.append("inline code changed")
    if extract_urls(original) != extract_urls(compressed):
        errors.append("URLs or markdown links changed")

    if original.startswith("---\n"):
        orig_match = re.match(r"(?s)^---\n.*?\n---\n?", original)
        comp_match = re.match(r"(?s)^---\n.*?\n---\n?", compressed)
        if (orig_match is None) != (comp_match is None):
            errors.append("frontmatter changed")
        elif orig_match and comp_match and orig_match.group(0) != comp_match.group(0):
            errors.append("frontmatter changed")

    return not errors, errors


def compress_with_anthropic(text: str, mode: str, model: str, api_key: Optional[str], max_tokens: int) -> str:
    try:
        import anthropic
    except Exception as exc:
        raise CompressionError(
            "Anthropic backend requested but `anthropic` package not installed. Install with: pip install anthropic"
        ) from exc

    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise CompressionError("Anthropic backend requested but ANTHROPIC_API_KEY is missing.")

    protected_text, protected = protect_spans(text)

    mode_detail = {
        "lite": "Keep full sentences. Remove filler and hedging. Stay concise and clear.",
        "full": "Drop articles where natural. Fragments allowed. Use short synonyms.",
        "ultra": "Maximum safe terseness. Fragments preferred. Abbreviate only when obvious.",
    }[mode]

    prompt = (
        "Rewrite the following natural-language prompt chunk into terse caveman style.\n"
        "Preserve all meaning.\n"
        "Return only the rewritten chunk.\n"
        "Do not add commentary, labels, fences, or quotes.\n"
        "Any placeholder like __CAVEMAN_PROTECTED_N__ must be copied back exactly.\n"
        f"Mode: {mode}. {mode_detail}\n"
        "Preserve line breaks when they matter for bullets or list structure.\n\n"
        "Chunk:\n"
        f"{protected_text}"
    )

    client = anthropic.Anthropic(api_key=key)
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )

    pieces: List[str] = []
    for block in getattr(message, "content", []):
        if getattr(block, "type", None) == "text":
            pieces.append(block.text)

    result = "".join(pieces).strip()
    if not result:
        raise CompressionError("Anthropic returned empty content.")

    return restore_spans(result, protected)


def compress_paragraph(
    text: str,
    mode: str,
    backend: str,
    anthropic_model: str,
    anthropic_api_key: Optional[str],
    fallback_local: bool,
) -> str:
    if backend == "local":
        return compress_paragraph_local(text, mode)

    if backend == "anthropic":
        stripped = text.strip("\n")
        if not stripped.strip():
            return text

        lines = stripped.splitlines()
        out_lines: List[str] = []
        for line in lines:
            if not line.strip():
                out_lines.append("")
                continue

            bullet_match = BULLET_RE.match(line)
            if bullet_match:
                prefix, body = bullet_match.groups()
                try:
                    out_lines.append(prefix + compress_with_anthropic(
                        body,
                        mode=mode,
                        model=anthropic_model,
                        api_key=anthropic_api_key,
                        max_tokens=max(128, min(1024, len(body) // 2 + 128)),
                    ))
                except Exception:
                    if not fallback_local:
                        raise
                    out_lines.append(prefix + compress_sentence_local(body, mode))
                continue

            quote_match = BLOCKQUOTE_RE.match(line)
            if quote_match:
                marker = quote_match.group(0)
                body = line[len(marker):]
                try:
                    out_lines.append(marker + compress_with_anthropic(
                        body,
                        mode=mode,
                        model=anthropic_model,
                        api_key=anthropic_api_key,
                        max_tokens=max(128, min(1024, len(body) // 2 + 128)),
                    ))
                except Exception:
                    if not fallback_local:
                        raise
                    out_lines.append(marker + compress_sentence_local(body, mode))
                continue

            try:
                out_lines.append(compress_with_anthropic(
                    line,
                    mode=mode,
                    model=anthropic_model,
                    api_key=anthropic_api_key,
                    max_tokens=max(128, min(1024, len(line) // 2 + 128)),
                ))
            except Exception:
                if not fallback_local:
                    raise
                out_lines.append(compress_sentence_local(line, mode))

        result = "\n".join(out_lines)
        if text.endswith("\n"):
            result += "\n"
        return result

    raise CompressionError(f"Unknown backend: {backend}")


def compress_text(
    text: str,
    mode: str,
    backend: str,
    anthropic_model: str,
    anthropic_api_key: Optional[str],
    fallback_local: bool,
) -> str:
    segments = segment_text(text)
    out: List[str] = []
    for seg in segments:
        if seg.kind == "preserve":
            out.append(seg.text)
        else:
            out.append(compress_paragraph(
                seg.text,
                mode=mode,
                backend=backend,
                anthropic_model=anthropic_model,
                anthropic_api_key=anthropic_api_key,
                fallback_local=fallback_local,
            ))
    return "".join(out)


def make_overlay(agent: str, mode: str) -> str:
    mode_details = {
        "lite": (
            "Keep full sentences. Remove filler, pleasantries, hedging. Stay concise and professional."
        ),
        "full": (
            "Drop articles where natural. Fragments OK. Use short synonyms. Classic caveman compression."
        ),
        "ultra": (
            "Maximum safe terseness. Fragments preferred. Abbreviate common technical terms only when obvious. Use arrows only if clarity stays intact."
        ),
    }
    detail = mode_details[mode]
    label = agent.upper()

    if agent == "claude":
        preface = "Visible-answer style only. Keep reasoning depth unchanged. Compress outward wording, not thought."
    else:
        preface = "Use this as response-style guidance. Keep reasoning quality unchanged; only compress visible wording."

    return (
        f"# {label} caveman overlay ({mode})\n\n"
        f"{preface}\n\n"
        "Respond terse like smart caveman. All technical substance stay. Only fluff die.\n\n"
        "Rules:\n"
        f"- {detail}\n"
        "- Drop filler, pleasantries, and hedging.\n"
        "- Preserve technical terms, identifiers, file paths, URLs, commands, version numbers, and code exactly.\n"
        "- Prefer pattern: [thing] [action] [reason]. [next step].\n"
        "- Do not pad with enthusiasm or throat-clearing.\n"
        "- If task is safety-critical, destructive, irreversible, or user seems confused, switch to normal clear prose for that part. Resume terse mode after.\n"
        "- Do not reduce hidden reasoning depth. Reduce visible output only.\n\n"
        "Stop condition:\n"
        '- If explicitly told "stop caveman" or "normal mode", revert to normal prose.\n'
    )


def make_combined_prompt(compressed_text: str, agent: str, mode: str) -> str:
    return compressed_text.rstrip() + "\n\n---\n\n" + make_overlay(agent, mode) + "\n"


def process_file(
    path: Path,
    mode: str,
    backend: str,
    anthropic_model: str,
    anthropic_api_key: Optional[str],
    fallback_local: bool,
    write: bool,
    out_dir: Optional[Path],
    emit_overlays: bool,
    combine_for: Sequence[str],
) -> Report:
    ok, reason = should_process(path)
    if not ok:
        return Report(
            path=str(path),
            wrote=False,
            backend=backend,
            backup=None,
            output=None,
            original_chars=0,
            compressed_chars=0,
            original_words=0,
            compressed_words=0,
            char_savings_pct=0.0,
            word_savings_pct=0.0,
            validation_ok=False,
            skipped_reason=reason,
        )

    original = path.read_text(encoding="utf-8")
    compressed = compress_text(
        original,
        mode=mode,
        backend=backend,
        anthropic_model=anthropic_model,
        anthropic_api_key=anthropic_api_key,
        fallback_local=fallback_local,
    )
    validation_ok, _errors = validate_preservation(original, compressed)

    original_chars = len(original)
    compressed_chars = len(compressed)
    original_words = word_count(original)
    compressed_words = word_count(compressed)
    char_savings_pct = 0.0 if original_chars == 0 else ((original_chars - compressed_chars) / original_chars) * 100.0
    word_savings_pct = 0.0 if original_words == 0 else ((original_words - compressed_words) / original_words) * 100.0

    wrote = False
    backup_path: Optional[Path] = None
    output_path: Optional[Path] = None

    if validation_ok:
        target_dir = out_dir if out_dir else path.parent
        target_dir.mkdir(parents=True, exist_ok=True)

        if write:
            backup_path = insert_before_suffix(path, ".original")
            if not backup_path.exists():
                backup_path.write_text(original, encoding="utf-8")
            else:
                insert_before_suffix(path, ".original.backup").write_text(original, encoding="utf-8")
            output_path = path
            path.write_text(compressed, encoding="utf-8")
            wrote = True
        elif out_dir:
            output_path = target_dir / path.name
            output_path.write_text(compressed, encoding="utf-8")
            wrote = True

        if emit_overlays:
            overlay_root = out_dir if out_dir else path.parent
            overlay_dir = overlay_root / "caveman-overlays"
            overlay_dir.mkdir(parents=True, exist_ok=True)

            for agent in combine_for:
                (overlay_dir / f"{path.stem}.overlay.{agent}.{mode}.md").write_text(
                    make_overlay(agent, mode),
                    encoding="utf-8",
                )
                (overlay_dir / f"{path.stem}.combined.{agent}.{mode}.md").write_text(
                    make_combined_prompt(compressed, agent, mode),
                    encoding="utf-8",
                )

    return Report(
        path=str(path),
        wrote=wrote,
        backend=backend,
        backup=str(backup_path) if backup_path else None,
        output=str(output_path) if output_path else None,
        original_chars=original_chars,
        compressed_chars=compressed_chars,
        original_words=original_words,
        compressed_words=compressed_words,
        char_savings_pct=char_savings_pct,
        word_savings_pct=word_savings_pct,
        validation_ok=validation_ok,
        skipped_reason=None if validation_ok else "validation failed",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compress prompt and memory files into caveman-style prose while preserving code/URLs/headings as safely as possible."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    compress = sub.add_parser("compress", help="Compress one or more prompt/memory files")
    compress.add_argument("files", nargs="+", help="Files to compress")
    compress.add_argument("--mode", choices=["lite", "full", "ultra"], default="full", help="Compression aggressiveness")
    compress.add_argument("--backend", choices=["local", "anthropic"], default="local", help="Compression backend")
    compress.add_argument("--anthropic-model", default="claude-sonnet-4-6", help="Anthropic model ID when --backend anthropic")
    compress.add_argument("--anthropic-api-key", help="Anthropic API key (otherwise uses ANTHROPIC_API_KEY)")
    compress.add_argument("--no-fallback-local", action="store_true", help="Disable fallback to local compression if Anthropic call fails")
    compress.add_argument("--write", action="store_true", help="Overwrite original files and create .original backups")
    compress.add_argument("--out-dir", type=Path, help="Write compressed copies to this directory instead of printing only")
    compress.add_argument("--emit-overlays", action="store_true", help="Emit Claude/generic overlay instruction files beside output")
    compress.add_argument("--agents", nargs="*", default=["claude", "generic", "gemini", "openai"], choices=["claude", "generic", "gemini", "openai"], help="Which overlay files to emit")
    compress.add_argument("--json-report", action="store_true", help="Print machine-readable JSON report")

    overlays = sub.add_parser("emit-overlays", help="Emit overlay instruction files only")
    overlays.add_argument("--out-dir", type=Path, required=True, help="Directory to write overlay files to")
    overlays.add_argument("--mode", choices=["lite", "full", "ultra"], default="full")
    overlays.add_argument("--agents", nargs="*", default=["claude", "generic", "gemini", "openai"], choices=["claude", "generic", "gemini", "openai"])

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "emit-overlays":
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for agent in args.agents:
            path = args.out_dir / f"caveman-overlay.{agent}.{args.mode}.md"
            path.write_text(make_overlay(agent, args.mode), encoding="utf-8")
            print(f"Wrote {path}")
        return 0

    reports: List[Report] = []
    for file_str in args.files:
        path = Path(file_str).expanduser().resolve()
        report = process_file(
            path=path,
            mode=args.mode,
            backend=args.backend,
            anthropic_model=args.anthropic_model,
            anthropic_api_key=args.anthropic_api_key,
            fallback_local=not args.no_fallback_local,
            write=args.write,
            out_dir=args.out_dir,
            emit_overlays=args.emit_overlays,
            combine_for=args.agents,
        )
        reports.append(report)

    if args.json_report:
        print(json.dumps([dataclasses.asdict(r) for r in reports], indent=2))
    else:
        for r in reports:
            if r.skipped_reason:
                print(f"SKIP  {r.path}: {r.skipped_reason}")
                continue
            status = "OK" if r.validation_ok else "FAIL"
            print(
                f"{status}  {r.path} | backend={r.backend} | chars {r.original_chars}->{r.compressed_chars} "
                f"({r.char_savings_pct:.1f}% saved) | words {r.original_words}->{r.compressed_words} "
                f"({r.word_savings_pct:.1f}% saved)"
            )
            if r.backup:
                print(f"      backup: {r.backup}")
            if r.output and r.output != r.path:
                print(f"      output: {r.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
