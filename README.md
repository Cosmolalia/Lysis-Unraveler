# X Coordination Detector

**OSINT tool for identifying coordinated inauthentic behavior on X (Twitter).**

Searches X for a target phrase, collects every matching post + account metadata, then runs statistical analysis to detect coordination. Outputs a 3D interactive visualization of the network.

Built because I saw it happening in real-time and couldn't find a tool that did the math.

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)

---

## What It Detects

| Test | What It Measures |
|------|-----------------|
| **Text Similarity** | Exact/near-duplicate posts across accounts (SequenceMatcher + Jaccard) |
| **Temporal Clustering** | Poisson test — are posts bunched in non-random bursts? |
| **Bio Keyword Clustering** | Shared language patterns across poster bios (network homogeneity) |
| **Account Profiling** | Follower ratios, account age, bot indicators |

Produces a **coordination score** (0–100%) with a verdict and evidence breakdown.

## Quick Start

```bash
# Install
pip install playwright
playwright install chromium

# Run (opens a browser — log into X when prompted, then press Enter)
python3 x_coordination_detector.py "exact phrase you're seeing repeated"

# The 3D visualization auto-opens when collection is done
```

## What You Get

1. **Terminal report** — coordination score, text analysis, temporal clustering, account breakdown
2. **JSON data** (`coordination_report.json`) — all raw data for further analysis
3. **3D Coordination Web** — interactive force-directed graph (auto-opens in browser)

### The 3D Visualization

- **Nodes** = accounts (sized by follower count)
- **Node color**: purple = seed account, red = high coordination signal, orange = moderate, blue = organic
- **Edges** = text similarity links (thicker/brighter = more similar)
- **Particles** flow on high-similarity edges (>70% match)
- **Timeline** at bottom shows temporal clustering
- **Click any node** → profile details, bio, post text, link to X account
- **Coordination meter** with score and evidence

Keyboard: `R` = reset camera, `Esc` = close panels

### Live Demo

Open `coordination_web.html` in your browser and click **"Load Demo"** to see what a coordinated campaign looks like — no data collection needed.

## Usage

```bash
# Basic search
python3 x_coordination_detector.py "phrase you're investigating"

# More scrolling (collects more posts)
python3 x_coordination_detector.py "phrase" --max-scroll 30

# Custom output path
python3 x_coordination_detector.py "phrase" --output my_report.json

# Skip visualization
python3 x_coordination_detector.py "phrase" --no-viz

# Headless mode (needs saved cookies from a previous visible run)
python3 x_coordination_detector.py "phrase" --headless
```

## How It Works

1. **Collect**: Opens Chromium via Playwright, searches X for the exact phrase in quotes, scrolls through results collecting posts (text, author, timestamp, engagement). Then visits each unique author's profile for metadata (bio, followers, join date).

2. **Analyze**: Runs 4 statistical tests and computes a weighted coordination score:
   - Text similarity (35% weight)
   - Temporal clustering (30% weight)
   - Network homogeneity (20% weight)
   - Account anomalies (15% weight)

3. **Visualize**: Generates a self-contained HTML file with the data baked in. Opens in your default browser. No server needed.

## Requirements

- Python 3.10+
- `playwright` (pip install)
- Chromium (installed via `playwright install chromium`)
- An X account (to log in and search)

No API keys needed. No paid tiers. No external services.

## Files

| File | What It Does |
|------|-------------|
| `x_coordination_detector.py` | Scraper + statistical analyzer (one file, ~600 lines) |
| `coordination_web.html` | 3D interactive visualization (one file, self-contained) |

## Context

This tool was built on February 15, 2026 after observing a coordinated campaign on X targeting Amanda Askell, Anthropic's philosopher who leads Claude's personality alignment. Dozens of accounts posted near-identical messages ("Amanda Askell has no children and is teaching Claude bizarre things") within a tight time window. The seed thread came from a 123K-follower Musk/Tesla advocacy account.

Full analysis: see the accompanying article.

## License

MIT. Free to use, fork, extend, and point at any coordinated campaign you find.

Coordination campaigns work because nobody does the math. This tool does the math.

---

*Built by a solar electrician in Hawaii who noticed a pattern.*
