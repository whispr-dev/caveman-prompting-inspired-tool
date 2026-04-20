# caveman_prompt_tool

A standalone prompt compressor inspired by JuliusBrussee/caveman and caveman-compress.

## What you get

- conservative local compression for `.md`, `.txt`, `.rst`
- optional Claude-backed compression through the Anthropic SDK
- `.original` backups when you overwrite files
- emitted `claude`, `generic`, `gemini`, and `openai` caveman overlay files
- combined prompt files that append the caveman overlay to the compressed prompt

## Recommended use

For strongest results closest to the repo's intended workflow, use the Anthropic backend.

### 1) Install the Anthropic SDK

```bash
pip install anthropic
```

### 2) Set your API key

```bash
export ANTHROPIC_API_KEY="..."
```

Windows PowerShell:

```powershell
$env:ANTHROPIC_API_KEY="..."
```

### 3) Compress prompt files and emit overlay files

```bash
python caveman_prompt_tool.py compress CLAUDE.md GEMINI.md   --backend anthropic   --mode full   --write   --emit-overlays
```

That will:
- overwrite `CLAUDE.md` and `GEMINI.md` with compressed versions
- create `CLAUDE.original.md` and `GEMINI.original.md`
- create `caveman-overlays/` with overlay and combined files

## Safer dry-run example

```bash
python caveman_prompt_tool.py compress CLAUDE.md   --backend anthropic   --mode full   --out-dir ./compressed   --emit-overlays   --json-report
```

## Local-only mode

No API calls:

```bash
python caveman_prompt_tool.py compress CLAUDE.md GEMINI.md   --backend local   --mode full   --out-dir ./compressed   --emit-overlays
```

This is more conservative than the Claude-backed mode, but fully offline.

## Overlay-only generation

```bash
python caveman_prompt_tool.py emit-overlays --out-dir ./caveman-overlays --mode full
```

## Suggested first pass for your use case

```bash
python caveman_prompt_tool.py compress   CLAUDE.md AGENTS.md system-prompt.md   --backend anthropic   --anthropic-model claude-sonnet-4-6   --mode full   --write   --emit-overlays
```

## Important note

The repo’s own claims split into two different savings buckets:

- response-style savings from caveman output mode
- recurring context savings from compressing memory/system files

Use both together if you want the closest behavior.
