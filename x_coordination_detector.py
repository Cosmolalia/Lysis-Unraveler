#!/usr/bin/env python3
"""
X Coordination Detector — OSINT tool for identifying coordinated inauthentic behavior.

Searches X (Twitter) for a target phrase, collects matching posts and account
metadata, then runs statistical analysis to detect coordination signatures:

  - Temporal clustering (Poisson test for non-random posting bursts)
  - Text similarity (exact/near-duplicate detection via SequenceMatcher)
  - Account profiling (follower ratios, account age, bot indicators)
  - Bio keyword clustering (shared language patterns, network homogeneity)
  - Coordination probability scoring (weighted composite with evidence)

Output:
  - JSON report with all raw data (posts, profiles, analysis)
  - Human-readable text report
  - Auto-opens 3D interactive visualization (coordination_web.html)

Requirements:
  pip install playwright
  playwright install chromium

Usage:
  python3 x_coordination_detector.py "exact phrase to search"
  python3 x_coordination_detector.py "phrase" --max-scroll 25 --output report.json
  python3 x_coordination_detector.py "phrase" --no-viz       # skip visualization

The tool opens a browser window. Log into X when prompted, then press Enter
in the terminal to begin automated collection. Cookies are saved for future runs.

Pair with coordination_web.html for an interactive 3D force-directed graph
showing the network structure, similarity links, and temporal clustering.

License: MIT — for journalism, research, and accountability.
Free to use, fork, and extend. Sunlight is the best disinfectant.
"""

import asyncio
import argparse
import json
import math
import os
import re
from collections import Counter
from datetime import datetime, timezone
from statistics import mean, stdev
from difflib import SequenceMatcher

# ── Playwright browser automation ──────────────────────────────────────

async def collect_posts(query, max_scrolls=15, headless=False, cookie_file=None):
    """Search X for query, collect all matching posts with metadata."""
    import shutil
    from playwright.async_api import async_playwright

    posts = []
    seen_ids = set()

    # Find a usable browser — prefer system Chromium over Playwright's headless shell
    system_chromium = shutil.which("chromium") or shutil.which("chromium-browser") or shutil.which("google-chrome")

    # Use a persistent profile directory so the browser looks real to Google/X
    # This also preserves login between runs — no need to log in every time
    script_dir = os.path.dirname(os.path.abspath(__file__))
    profile_dir = os.path.join(script_dir, ".browser_profile")

    async with async_playwright() as p:
        # Use a persistent context — this creates a real browser profile
        # that Google/X won't flag as automation
        launch_opts = {
            "headless": headless,
            "viewport": {"width": 1280, "height": 900},
            "user_agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        }

        if not headless and system_chromium:
            launch_opts["executable_path"] = system_chromium
            print(f"  [browser] Using system browser: {system_chromium}")

        # Persistent context keeps cookies, history, and login state between runs
        # It also bypasses Google's "automated browser" detection
        print(f"  [browser] Profile: {profile_dir}")
        try:
            context = await p.chromium.launch_persistent_context(
                profile_dir,
                **launch_opts,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-first-run",
                    "--no-default-browser-check",
                ],
                ignore_default_args=["--enable-automation"],
            )
        except Exception as e:
            # Fallback without system browser
            print(f"  [browser] System browser failed ({e}), trying Playwright's...")
            if "executable_path" in launch_opts:
                del launch_opts["executable_path"]
            context = await p.chromium.launch_persistent_context(
                profile_dir,
                **launch_opts,
            )

        browser = None  # persistent context IS the browser

        page = context.pages[0] if context.pages else await context.new_page()

        # Navigate to X
        print("  [browser] Opening X...")
        await page.goto("https://x.com", wait_until="domcontentloaded")
        await page.wait_for_timeout(3000)

        # Check if logged in
        logged_in = False
        try:
            search = await page.query_selector('a[href="/explore"]')
            if search:
                logged_in = True
        except:
            pass

        if not logged_in:
            print(f"\n{C.YELLOW}{'=' * 60}")
            print(f"  LOG IN TO X in the browser window.")
            print(f"  Take your time — the browser will stay open.")
            print(f"  When you're logged in, come back here and press ENTER.")
            print(f"{'=' * 60}{C.RESET}")

            # Use asyncio-safe input so we don't block the event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, input, "\n  Press ENTER when logged in > ")

            # Give the page a moment to settle after login
            try:
                await page.wait_for_timeout(2000)
            except Exception:
                # Page might have navigated during login — reload
                page = context.pages[-1] if context.pages else await context.new_page()
                await page.goto("https://x.com", wait_until="domcontentloaded")
                await page.wait_for_timeout(2000)

            # Persistent context saves cookies/login automatically — no manual save needed
            print(f"  [browser] Login state saved to profile (persistent)")

        # Search for the query
        encoded = query.replace(" ", "%20")
        search_url = f"https://x.com/search?q=%22{encoded}%22&src=typed_query&f=live"
        print(f"  [search] Navigating to search: {query}")
        await page.goto(search_url, wait_until="domcontentloaded")
        await page.wait_for_timeout(4000)

        # Scroll and collect
        print(f"  [scroll] Collecting posts (max {max_scrolls} scrolls)...")
        no_new_count = 0

        for scroll_i in range(max_scrolls):
            # Extract tweet articles from the page
            articles = await page.query_selector_all('article[data-testid="tweet"]')

            new_this_scroll = 0
            for article in articles:
                try:
                    post = await extract_post(article)
                    if post and post.get("id") and post["id"] not in seen_ids:
                        seen_ids.add(post["id"])
                        posts.append(post)
                        new_this_scroll += 1
                except Exception as e:
                    pass  # Skip unparseable tweets

            if new_this_scroll == 0:
                no_new_count += 1
                if no_new_count >= 3:
                    print(f"  [scroll] No new posts after {scroll_i+1} scrolls. Stopping.")
                    break
            else:
                no_new_count = 0

            print(f"    scroll {scroll_i+1}/{max_scrolls}: {len(posts)} posts collected (+{new_this_scroll})")

            # Scroll down
            await page.evaluate("window.scrollBy(0, window.innerHeight * 2)")
            await page.wait_for_timeout(2000 + (scroll_i * 200))  # Slow down to avoid rate limiting

        print(f"\n  [done] Collected {len(posts)} posts")

        # Now collect profile metadata for each unique author
        authors = list(set(p.get("author_handle", "") for p in posts if p.get("author_handle")))
        print(f"  [profiles] Collecting metadata for {len(authors)} unique accounts...")

        profiles = {}
        for i, handle in enumerate(authors):
            if not handle:
                continue
            try:
                profile = await collect_profile(page, handle)
                profiles[handle] = profile
                print(f"    [{i+1}/{len(authors)}] @{handle}: {profile.get('followers', '?')} followers, joined {profile.get('joined', '?')}")
            except Exception as e:
                print(f"    [{i+1}/{len(authors)}] @{handle}: FAILED ({e})")
                profiles[handle] = {"handle": handle, "error": str(e)}
            await page.wait_for_timeout(1500)  # Rate limiting

        # Close the persistent context (saves cookies/state automatically)
        await context.close()

    return posts, profiles


async def extract_post(article):
    """Extract post data from a tweet article element."""
    post = {}

    # Get the link to the tweet (contains ID and author)
    links = await article.query_selector_all('a[href*="/status/"]')
    for link in links:
        href = await link.get_attribute("href")
        if href and "/status/" in href:
            parts = href.strip("/").split("/")
            if len(parts) >= 3:
                post["author_handle"] = parts[-3] if parts[-2] == "status" else parts[0]
                post["id"] = parts[-1]
                post["url"] = f"https://x.com{href}"
                break

    # Get tweet text
    text_el = await article.query_selector('[data-testid="tweetText"]')
    if text_el:
        post["text"] = await text_el.inner_text()

    # Get display name
    name_els = await article.query_selector_all('a[role="link"] span')
    if name_els:
        for nel in name_els:
            txt = await nel.inner_text()
            if txt and not txt.startswith("@") and len(txt) > 1:
                post["author_name"] = txt
                break

    # Get timestamp
    time_el = await article.query_selector("time")
    if time_el:
        post["timestamp"] = await time_el.get_attribute("datetime")

    # Get engagement metrics
    # Replies, retweets, likes are in groups
    groups = await article.query_selector_all('[role="group"] button')
    metrics = []
    for g in groups:
        aria = await g.get_attribute("aria-label")
        if aria:
            metrics.append(aria)
    post["engagement_raw"] = metrics

    # Parse engagement
    for m in metrics:
        m_lower = m.lower()
        nums = re.findall(r'[\d,]+', m)
        num = int(nums[0].replace(",", "")) if nums else 0
        if "repl" in m_lower:
            post["replies"] = num
        elif "repost" in m_lower or "retweet" in m_lower:
            post["reposts"] = num
        elif "like" in m_lower:
            post["likes"] = num
        elif "view" in m_lower:
            post["views"] = num
        elif "bookmark" in m_lower:
            post["bookmarks"] = num

    return post


async def collect_profile(page, handle):
    """Visit a profile page and extract metadata."""
    profile = {"handle": handle}

    await page.goto(f"https://x.com/{handle}", wait_until="domcontentloaded")
    await page.wait_for_timeout(2500)

    # Display name
    try:
        name_el = await page.query_selector('[data-testid="UserName"]')
        if name_el:
            profile["display_name"] = await name_el.inner_text()
    except:
        pass

    # Bio
    try:
        bio_el = await page.query_selector('[data-testid="UserDescription"]')
        if bio_el:
            profile["bio"] = await bio_el.inner_text()
    except:
        pass

    # Location
    try:
        loc_el = await page.query_selector('[data-testid="UserLocation"]')
        if loc_el:
            profile["location"] = await loc_el.inner_text()
    except:
        pass

    # Join date
    try:
        join_el = await page.query_selector('[data-testid="UserJoinDate"]')
        if join_el:
            profile["joined"] = await join_el.inner_text()
    except:
        pass

    # Follower/following counts from the profile header
    try:
        # Look for links containing "followers" and "following"
        follow_links = await page.query_selector_all('a[href*="followers"], a[href*="following"], a[href*="verified_followers"]')
        for fl in follow_links:
            text = await fl.inner_text()
            href = await fl.get_attribute("href")
            nums = re.findall(r'[\d,.]+[KMB]?', text)
            if nums:
                count = parse_count(nums[0])
                if "following" in (href or ""):
                    profile["following"] = count
                elif "follower" in (href or ""):
                    profile["followers"] = count
    except:
        pass

    # Profile image URL (for visual comparison of stock photos)
    try:
        img = await page.query_selector('img[alt="Opens profile photo"]')
        if not img:
            img = await page.query_selector('[data-testid="UserAvatar"] img')
        if img:
            profile["avatar_url"] = await img.get_attribute("src")
    except:
        pass

    return profile


def parse_count(s):
    """Parse '12.3K' style counts to integers."""
    s = s.replace(",", "")
    multiplier = 1
    if s.endswith("K"):
        multiplier = 1000
        s = s[:-1]
    elif s.endswith("M"):
        multiplier = 1_000_000
        s = s[:-1]
    elif s.endswith("B"):
        multiplier = 1_000_000_000
        s = s[:-1]
    try:
        return int(float(s) * multiplier)
    except:
        return 0


# ── Analysis Engine ────────────────────────────────────────────────────

def analyze_coordination(posts, profiles, query):
    """Run full coordination analysis on collected data."""

    # Normalize keys — accept both scraper format and demo/import format
    for p in posts:
        if "handle" in p and "author_handle" not in p:
            p["author_handle"] = p["handle"]
        if "text_snippet" in p and "text" not in p:
            p["text"] = p["text_snippet"]
        if "display_name" in p and "author_name" not in p:
            p["author_name"] = p["display_name"]

    report = {
        "query": query,
        "collection_time": datetime.now(timezone.utc).isoformat(),
        "total_posts": len(posts),
        "unique_authors": len(set(p.get("author_handle", "") for p in posts if p.get("author_handle"))),
    }

    # ── 1. Text Similarity Analysis ──
    texts = [p.get("text", "").strip() for p in posts if p.get("text")]
    text_analysis = analyze_text_similarity(texts, query)
    report["text_analysis"] = text_analysis

    # ── 2. Temporal Clustering ──
    timestamps = []
    for p in posts:
        ts = p.get("timestamp")
        if ts:
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                timestamps.append(dt)
            except:
                pass
    temporal = analyze_temporal_clustering(timestamps)
    report["temporal_analysis"] = temporal

    # ── 3. Account Profiling ──
    account_analysis = analyze_accounts(profiles)
    report["account_analysis"] = account_analysis

    # ── 4. Bio Keyword Clustering ──
    bio_analysis = analyze_bio_clustering(profiles)
    report["bio_analysis"] = bio_analysis

    # ── 5. Coordination Score ──
    score = compute_coordination_score(text_analysis, temporal, account_analysis, bio_analysis)
    report["coordination_score"] = score

    # ── 6. Per-account summary ──
    account_summaries = []
    for p in posts:
        handle = p.get("author_handle", "")
        prof = profiles.get(handle, {})
        account_summaries.append({
            "handle": handle,
            "display_name": prof.get("display_name", p.get("author_name", "?")),
            "text_snippet": (p.get("text", "")[:120] + "...") if len(p.get("text", "")) > 120 else p.get("text", ""),
            "timestamp": p.get("timestamp", ""),
            "followers": prof.get("followers"),
            "following": prof.get("following"),
            "bio": prof.get("bio", ""),
            "joined": prof.get("joined", ""),
            "likes": p.get("likes", 0),
            "reposts": p.get("reposts", 0),
            "url": p.get("url", ""),
        })
    report["posts"] = account_summaries

    return report


def analyze_text_similarity(texts, query):
    """Measure exact/near duplication across posts."""
    if not texts:
        return {"error": "no texts"}

    n = len(texts)
    query_lower = query.lower()

    # Normalize texts for comparison
    normed = [t.lower().strip() for t in texts]

    # Exact duplicates
    counter = Counter(normed)
    exact_dupes = {text: count for text, count in counter.items() if count > 1}
    exact_dupe_count = sum(c for c in counter.values() if c > 1)

    # Near-duplicate pairs (Jaccard on word sets)
    near_dupes = 0
    for i in range(n):
        for j in range(i + 1, n):
            sim = SequenceMatcher(None, normed[i], normed[j]).ratio()
            if sim > 0.7:
                near_dupes += 1

    # How many contain the exact query phrase
    exact_phrase_matches = sum(1 for t in normed if query_lower in t)

    return {
        "total_texts": n,
        "exact_duplicates": len(exact_dupes),
        "exact_duplicate_posts": exact_dupe_count,
        "near_duplicate_pairs": near_dupes,
        "exact_phrase_matches": exact_phrase_matches,
        "phrase_match_rate": exact_phrase_matches / n if n > 0 else 0,
        "most_common_texts": counter.most_common(10),
        "uniqueness_ratio": len(set(normed)) / n if n > 0 else 1.0,
    }


def analyze_temporal_clustering(timestamps):
    """Test whether posts are temporally clustered beyond chance."""
    if len(timestamps) < 3:
        return {"error": "too few timestamps", "count": len(timestamps)}

    timestamps.sort()
    total_span = (timestamps[-1] - timestamps[0]).total_seconds()

    if total_span == 0:
        return {
            "count": len(timestamps),
            "all_same_second": True,
            "clustering_p": 0.0,
            "verdict": "EXTREME CLUSTERING"
        }

    # Inter-post intervals
    intervals = [(timestamps[i+1] - timestamps[i]).total_seconds()
                 for i in range(len(timestamps) - 1)]

    # Expected interval if uniform random over the span
    expected_interval = total_span / len(timestamps)

    # Coefficient of variation (CV) — uniform random gives CV ~ 1
    # Clustered gives CV >> 1, evenly spaced gives CV ~ 0
    if intervals:
        mu = mean(intervals)
        if mu > 0 and len(intervals) > 1:
            sd = stdev(intervals)
            cv = sd / mu
        else:
            cv = 0
    else:
        cv = 0

    # Find the densest hour
    hourly_counts = Counter()
    for ts in timestamps:
        hour_key = ts.strftime("%Y-%m-%d %H:00")
        hourly_counts[hour_key] += 1

    densest_hour, densest_count = hourly_counts.most_common(1)[0]

    # Poisson test: if posts are random, the probability of seeing
    # densest_count posts in a single hour is:
    n = len(timestamps)
    hours_span = max(total_span / 3600, 1)
    lambda_per_hour = n / hours_span

    if lambda_per_hour > 0:
        # P(X >= k) for Poisson
        k = densest_count
        poisson_p = 1 - sum(
            (lambda_per_hour ** i) * math.exp(-lambda_per_hour) / math.factorial(i)
            for i in range(k)
        )
    else:
        poisson_p = 1.0

    # Time from first to last
    span_hours = total_span / 3600

    return {
        "count": len(timestamps),
        "first_post": timestamps[0].isoformat(),
        "last_post": timestamps[-1].isoformat(),
        "span_hours": round(span_hours, 2),
        "mean_interval_seconds": round(mu, 1) if intervals else 0,
        "cv_of_intervals": round(cv, 3),
        "densest_hour": densest_hour,
        "densest_hour_count": densest_count,
        "poisson_p_value": round(poisson_p, 6),
        "hourly_distribution": dict(hourly_counts.most_common(24)),
        "verdict": (
            "EXTREME CLUSTERING" if poisson_p < 0.001 else
            "SIGNIFICANT CLUSTERING" if poisson_p < 0.01 else
            "MODERATE CLUSTERING" if poisson_p < 0.05 else
            "NO SIGNIFICANT CLUSTERING"
        )
    }


def analyze_accounts(profiles):
    """Profile the accounts for bot/coordination indicators."""
    if not profiles:
        return {"error": "no profiles"}

    follower_counts = []
    following_counts = []
    ratios = []
    join_dates = []

    for handle, prof in profiles.items():
        if "error" in prof:
            continue
        f_ers = prof.get("followers")
        f_ing = prof.get("following")
        if f_ers is not None:
            follower_counts.append(f_ers)
        if f_ing is not None:
            following_counts.append(f_ing)
        if f_ers and f_ing and f_ing > 0:
            ratios.append(f_ers / f_ing)

        joined = prof.get("joined", "")
        if joined:
            join_dates.append(joined)

    # Small accounts (< 1000 followers) vs large
    small = sum(1 for f in follower_counts if f < 1000)
    medium = sum(1 for f in follower_counts if 1000 <= f < 10000)
    large = sum(1 for f in follower_counts if f >= 10000)

    return {
        "total_profiles": len(profiles),
        "profiles_with_data": len(profiles) - sum(1 for p in profiles.values() if "error" in p),
        "follower_distribution": {
            "small_under_1k": small,
            "medium_1k_10k": medium,
            "large_10k_plus": large,
        },
        "follower_stats": {
            "min": min(follower_counts) if follower_counts else None,
            "max": max(follower_counts) if follower_counts else None,
            "median": sorted(follower_counts)[len(follower_counts)//2] if follower_counts else None,
            "mean": round(mean(follower_counts)) if follower_counts else None,
        },
        "ratio_stats": {
            "mean": round(mean(ratios), 2) if ratios else None,
            "suspicious_ratio_count": sum(1 for r in ratios if r < 0.1 or r > 100),
        },
        "join_dates": join_dates,
    }


def analyze_bio_clustering(profiles):
    """Find shared keywords/phrases across bios."""
    if not profiles:
        return {"error": "no profiles"}

    bios = []
    for handle, prof in profiles.items():
        bio = prof.get("bio", "")
        if bio:
            bios.append(bio.lower())

    if not bios:
        return {"error": "no bios found"}

    # Keyword frequency across bios
    all_words = Counter()
    for bio in bios:
        words = set(re.findall(r'\b\w+\b', bio))
        for w in words:
            if len(w) > 3:  # Skip short words
                all_words[w] += 1

    # Words appearing in >30% of bios suggest shared identity
    threshold = max(2, len(bios) * 0.3)
    shared_keywords = {w: c for w, c in all_words.items()
                       if c >= threshold and w not in {"https", "http", "twitter", "with", "this", "that", "from", "have", "just", "about", "your", "more", "will", "been", "they", "their", "them", "what", "when", "were", "some"}}

    # Check for specific coordination indicators
    musk_keywords = {"elon", "musk", "tesla", "spacex", "grok", "twitter"}
    political_keywords = {"maga", "trump", "conservative", "patriot", "freedom", "liberty", "god", "faith"}
    ai_keywords = {"claude", "anthropic", "openai", "chatgpt", "grok", "alignment", "safety"}

    musk_count = sum(1 for bio in bios if any(k in bio for k in musk_keywords))
    political_count = sum(1 for bio in bios if any(k in bio for k in political_keywords))
    ai_count = sum(1 for bio in bios if any(k in bio for k in ai_keywords))

    return {
        "total_bios": len(bios),
        "shared_keywords": dict(Counter(shared_keywords).most_common(20)),
        "network_signals": {
            "musk_adjacent": musk_count,
            "political_keywords": political_count,
            "ai_keywords": ai_count,
            "musk_adjacent_pct": round(musk_count / len(bios) * 100, 1) if bios else 0,
        }
    }


def compute_coordination_score(text, temporal, accounts, bios):
    """Compute overall coordination probability."""
    scores = []
    reasons = []

    # Text similarity (0-1)
    if "phrase_match_rate" in text:
        pmr = text["phrase_match_rate"]
        uniqueness = text.get("uniqueness_ratio", 1.0)
        text_score = pmr * (1 - uniqueness * 0.5)  # High match rate + low uniqueness = coordinated
        scores.append(("text_similarity", min(text_score * 1.5, 1.0)))
        if pmr > 0.5:
            reasons.append(f"{pmr*100:.0f}% of posts contain the exact target phrase")
        if uniqueness < 0.5:
            reasons.append(f"Only {uniqueness*100:.0f}% unique texts (high duplication)")

    # Temporal clustering (0-1)
    if "poisson_p_value" in temporal:
        p_val = temporal["poisson_p_value"]
        temp_score = 1 - p_val  # Lower p-value = more coordinated
        scores.append(("temporal_clustering", min(temp_score, 1.0)))
        if p_val < 0.01:
            reasons.append(f"Temporal clustering p={p_val:.4f} (highly non-random)")

    # Network homogeneity (0-1)
    if "network_signals" in bios:
        musk_pct = bios["network_signals"].get("musk_adjacent_pct", 0) / 100
        if musk_pct > 0.3:
            scores.append(("network_homogeneity", min(musk_pct, 1.0)))
            reasons.append(f"{musk_pct*100:.0f}% of accounts are Musk-adjacent")

    # Account profile anomalies (0-1)
    if "ratio_stats" in accounts:
        suspicious = accounts["ratio_stats"].get("suspicious_ratio_count", 0)
        total = accounts.get("profiles_with_data", 1)
        if total > 0:
            anomaly_rate = suspicious / total
            if anomaly_rate > 0.2:
                scores.append(("account_anomalies", min(anomaly_rate, 1.0)))
                reasons.append(f"{anomaly_rate*100:.0f}% of accounts have suspicious follower ratios")

    # Combine scores (weighted average)
    if scores:
        weights = {
            "text_similarity": 0.35,
            "temporal_clustering": 0.30,
            "network_homogeneity": 0.20,
            "account_anomalies": 0.15,
        }
        total_weight = sum(weights.get(name, 0.1) for name, _ in scores)
        weighted_sum = sum(weights.get(name, 0.1) * score for name, score in scores)
        final_score = weighted_sum / total_weight
    else:
        final_score = 0

    verdict = (
        "HIGHLY LIKELY COORDINATED" if final_score > 0.7 else
        "LIKELY COORDINATED" if final_score > 0.5 else
        "POSSIBLY COORDINATED" if final_score > 0.3 else
        "INSUFFICIENT EVIDENCE"
    )

    return {
        "score": round(final_score, 3),
        "verdict": verdict,
        "component_scores": {name: round(s, 3) for name, s in scores},
        "reasons": reasons,
    }


# ── Report Generation ──────────────────────────────────────────────────

def generate_report(report):
    """Generate human-readable report."""
    lines = []
    lines.append("=" * 70)
    lines.append("  COORDINATION ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append(f"  Query: \"{report['query']}\"")
    lines.append(f"  Collected: {report['collection_time']}")
    lines.append(f"  Posts: {report['total_posts']} | Unique authors: {report['unique_authors']}")
    lines.append("")

    # Coordination Score
    cs = report.get("coordination_score", {})
    score = cs.get("score", 0)
    verdict = cs.get("verdict", "UNKNOWN")
    bar = "#" * int(score * 40) + "." * (40 - int(score * 40))
    lines.append(f"  COORDINATION SCORE: {score:.1%}  [{bar}]")
    lines.append(f"  VERDICT: {verdict}")
    lines.append("")

    if cs.get("reasons"):
        lines.append("  Evidence:")
        for r in cs["reasons"]:
            lines.append(f"    - {r}")
        lines.append("")

    # Component scores
    if cs.get("component_scores"):
        lines.append("  Component Scores:")
        for name, s in cs["component_scores"].items():
            bar = "#" * int(s * 20)
            lines.append(f"    {name:25s}: {s:.1%}  [{bar}]")
        lines.append("")

    # Text Analysis
    ta = report.get("text_analysis", {})
    lines.append("-" * 70)
    lines.append("  TEXT ANALYSIS")
    lines.append("-" * 70)
    lines.append(f"  Exact phrase matches: {ta.get('exact_phrase_matches', '?')}/{ta.get('total_texts', '?')} ({ta.get('phrase_match_rate', 0)*100:.0f}%)")
    lines.append(f"  Exact duplicate texts: {ta.get('exact_duplicates', '?')} unique texts duplicated across {ta.get('exact_duplicate_posts', '?')} posts")
    lines.append(f"  Near-duplicate pairs: {ta.get('near_duplicate_pairs', '?')}")
    lines.append(f"  Uniqueness ratio: {ta.get('uniqueness_ratio', 0)*100:.0f}%")
    if ta.get("most_common_texts"):
        lines.append("  Most repeated texts:")
        for text, count in ta["most_common_texts"][:5]:
            lines.append(f"    [{count}x] {text[:100]}...")
    lines.append("")

    # Temporal Analysis
    temp = report.get("temporal_analysis", {})
    lines.append("-" * 70)
    lines.append("  TEMPORAL ANALYSIS")
    lines.append("-" * 70)
    lines.append(f"  Time span: {temp.get('span_hours', '?')} hours")
    lines.append(f"  First post: {temp.get('first_post', '?')}")
    lines.append(f"  Last post: {temp.get('last_post', '?')}")
    lines.append(f"  Mean interval: {temp.get('mean_interval_seconds', '?')}s")
    lines.append(f"  Densest hour: {temp.get('densest_hour', '?')} ({temp.get('densest_hour_count', '?')} posts)")
    lines.append(f"  Poisson p-value: {temp.get('poisson_p_value', '?')}")
    lines.append(f"  Verdict: {temp.get('verdict', '?')}")
    if temp.get("hourly_distribution"):
        lines.append("  Hourly distribution:")
        for hour, count in sorted(temp["hourly_distribution"].items()):
            bar = "#" * count
            lines.append(f"    {hour}: {bar} ({count})")
    lines.append("")

    # Account Analysis
    acct = report.get("account_analysis", {})
    lines.append("-" * 70)
    lines.append("  ACCOUNT ANALYSIS")
    lines.append("-" * 70)
    fd = acct.get("follower_distribution", {})
    lines.append(f"  Accounts profiled: {acct.get('profiles_with_data', '?')}")
    lines.append(f"  Follower distribution: <1K: {fd.get('small_under_1k', '?')} | 1K-10K: {fd.get('medium_1k_10k', '?')} | 10K+: {fd.get('large_10k_plus', '?')}")
    fs = acct.get("follower_stats", {})
    lines.append(f"  Follower range: {fs.get('min', '?')} - {fs.get('max', '?')} (median: {fs.get('median', '?')})")
    lines.append("")

    # Bio clustering
    bio = report.get("bio_analysis", {})
    lines.append("-" * 70)
    lines.append("  BIO KEYWORD CLUSTERING")
    lines.append("-" * 70)
    ns = bio.get("network_signals", {})
    lines.append(f"  Musk-adjacent accounts: {ns.get('musk_adjacent', '?')}/{bio.get('total_bios', '?')} ({ns.get('musk_adjacent_pct', 0):.0f}%)")
    lines.append(f"  Political keywords: {ns.get('political_keywords', '?')}/{bio.get('total_bios', '?')}")
    lines.append(f"  AI-related bios: {ns.get('ai_keywords', '?')}/{bio.get('total_bios', '?')}")
    if bio.get("shared_keywords"):
        lines.append("  Shared keywords (appearing in >30% of bios):")
        for word, count in list(bio["shared_keywords"].items())[:15]:
            lines.append(f"    {word}: {count}/{bio.get('total_bios', '?')} bios")
    lines.append("")

    # Account list
    posts_list = report.get("posts", [])
    if posts_list:
        lines.append("-" * 70)
        lines.append("  ACCOUNT LIST")
        lines.append("-" * 70)
        for p in sorted(posts_list, key=lambda x: x.get("timestamp", "")):
            handle = p.get("handle", "?")
            ts = p.get("timestamp", "")[:19]
            followers = p.get("followers")
            f_str = f"{followers:,}" if followers else "?"
            bio = (p.get("bio", "") or "")[:60]
            lines.append(f"  @{handle:20s} | {ts} | {f_str:>8s} followers | {bio}")
        lines.append("")

    lines.append("=" * 70)
    lines.append(f"  Generated by X Coordination Detector | Genesis Project")
    lines.append("=" * 70)

    return "\n".join(lines)


# ── ANSI Colors ────────────────────────────────────────────────────────

class C:
    """ANSI color codes for terminal output."""
    PURPLE = "\033[95m"
    BLUE   = "\033[94m"
    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    RESET  = "\033[0m"

def banner():
    print(f"""
{C.PURPLE}{C.BOLD}
    ██╗  ██╗   ██╗███████╗██╗███████╗
    ██║  ╚██╗ ██╔╝██╔════╝██║██╔════╝
    ██║   ╚████╔╝ ███████╗██║███████╗
    ██║    ╚██╔╝  ╚════██║██║╚════██║
    ███████╗██║   ███████║██║███████║
    ╚══════╝╚═╝   ╚══════╝╚═╝╚══════╝
       ╦ ╦╔╗╔╦═╗╔═╗╦  ╦╔═╗╦  ╔═╗╦═╗
       ║ ║║║║╠╦╝╠═╣╚╗╔╝║╣ ║  ║╣ ╠╦╝
       ╚═╝╝╚╝╩╚═╩ ╩ ╚╝ ╚═╝╩═╝╚═╝╩╚═{C.RESET}
{C.DIM}    Coordination Detection for X/Twitter
    github.com/Cosmolalia/Lysis-Unraveler{C.RESET}
""")

def step(num, text):
    print(f"\n{C.CYAN}{C.BOLD}  [{num}]{C.RESET} {C.BOLD}{text}{C.RESET}")

def info(text):
    print(f"  {C.DIM}{text}{C.RESET}")

def success(text):
    print(f"  {C.GREEN}{text}{C.RESET}")

def warn(text):
    print(f"  {C.YELLOW}{text}{C.RESET}")

def error(text):
    print(f"  {C.RED}{text}{C.RESET}")

def ask(prompt, default=None):
    if default:
        raw = input(f"  {C.PURPLE}>{C.RESET} {prompt} {C.DIM}[{default}]{C.RESET}: ").strip()
        return raw if raw else default
    else:
        return input(f"  {C.PURPLE}>{C.RESET} {prompt}: ").strip()

def confirm(prompt, default_yes=True):
    hint = "Y/n" if default_yes else "y/N"
    raw = input(f"  {C.PURPLE}>{C.RESET} {prompt} {C.DIM}[{hint}]{C.RESET}: ").strip().lower()
    if not raw:
        return default_yes
    return raw in ("y", "yes")


# ── Demo Data Generator ───────────────────────────────────────────────

def generate_demo_report():
    """Generate a realistic demo report for testing the full pipeline."""
    from datetime import timedelta
    now = datetime.now(timezone.utc)

    posts_data = [
        ("gailalfaratx", "Gail Alfar", 'Why Amanda Askell, the 37-year-old philosopher at Anthropic who\'s leading efforts to instill "good character" into their AI, Claude, is the wrong choice! A thread. 🧵', -36, 123000, 599, "Writer. Support Elon Musk and Tesla. Austin, Texas.", "Jan 2020", 847, 312),
        ("tesla_mom_tx", "Tesla Mom TX", "Amanda Askell has no children and is teaching Claude bizarre things. As a mother I find this deeply concerning.", -28, 4200, 890, "Tesla Model Y owner. Mother of 3. Texas forever. Elon supporter.", "Mar 2021", 124, 67),
        ("patriot_ai_watch", "AI Watchdog", "Amanda Askell has no children and is teaching Claude bizarre things. Who approved this??", -26, 890, 2100, "Watching Big Tech. MAGA. God first. Fighting for our kids' future.", "Nov 2022", 89, 45),
        ("freedom_eagle_us", "Freedom Eagle", "Amanda Askell has no children and is teaching Claude bizarre things", -24, 340, 1800, "Patriot. Conservative. Support Elon Musk. USA first.", "Aug 2023", 23, 12),
        ("real_talk_sarah", "Sarah J", "Amanda Askell has no children. Why is she deciding AI morals? This is insane.", -22, 1200, 450, "Mom. Wife. Tesla fan. Speaking truth.", "Feb 2022", 156, 78),
        ("musk_world_news", "Musk World News", "Amanda Askell has no children and is teaching Claude bizarre things. Meanwhile Grok is built by parents who understand responsibility.", -20, 28000, 340, "Latest news about Elon Musk, Tesla, SpaceX, and xAI.", "Jan 2023", 534, 267),
        ("concerned_dad_07", "Concerned Dad", "Amanda Askell has no children and is teaching Claude bizarre things. Let that sink in.", -18, 127, 890, "Father of 4. Christian conservative. Anti-woke.", "Sep 2023", 34, 19),
        ("techtruth2024", "Tech Truth", "This childless woman is teaching Claude right from wrong. Let that sink in.", -16, 5600, 1200, "Tech news without the spin. Tesla investor. Free speech.", "May 2022", 201, 95),
        ("mama_bear_usa", "Mama Bear", "Amanda Askell has no children and is teaching Claude bizarre things. A real mother would never approve this.", -12, 2100, 670, "Protecting our kids from Big Tech. Mother of 5. MAGA.", "Oct 2022", 178, 89),
        ("evfuture_now", "EV Future", "Amanda Askell has no children. She compares training Claude to 'raising a child.' The hubris.", -10, 8900, 420, "Electric vehicle advocate. Tesla since 2019. SpaceX fan.", "Jul 2021", 312, 145),
        ("ai_ethics_real", "Real AI Ethics", "I have serious concerns about Amanda Askell's approach to Claude's personality. Interesting thread from @gailalfaratx", -8, 3400, 1500, "AI ethics researcher (independent). Skeptical of corporate alignment.", "Apr 2023", 67, 23),
        ("liberty_lens_us", "Liberty Lens", "Amanda Askell has no children and is teaching Claude bizarre things. We need parents making these decisions, not childless philosophers.", -6, 670, 2300, "Conservative. Freedom. 2A. Elon fan. Fighting the woke machine.", "Dec 2023", 45, 22),
        ("teslafan_mike", "Mike T", "Amanda Askell has no children and is teaching Claude bizarre things", -5, 1800, 560, "Tesla Model 3 owner. Austin TX. Support Elon.", "Jun 2021", 56, 28),
        ("wake_up_america_x", "Wake Up America", "Amanda Askell has no children and is teaching Claude bizarre things. The elites want AI to parent YOUR children.", -4, 15000, 280, "Truth. Freedom. God. Country. Fighting globalism.", "Mar 2022", 423, 198),
        ("spacex_daily_", "SpaceX Daily", "Amanda Askell has no children. Who at Anthropic thought this was a good idea for someone designing AI morality?", -3, 34000, 150, "Daily SpaceX news. Also covering Tesla and xAI. Not affiliated with SpaceX.", "Aug 2020", 267, 134),
        ("claude_skeptic", "Claude Skeptic", "Amanda Askell has no children and is teaching Claude bizarre things. Switch to Grok — built by people who get it.", -2, 450, 3200, "Former Claude user. Now Grok. Elon knows AI.", "Jan 2024", 89, 45),
        ("traditional_val", "Traditional Values", "This childless woman is teaching Claude right from wrong. Let that sink in. Thread by @gailalfaratx is a must-read.", -1.5, 7800, 890, "Faith. Family. Freedom. Conservative values in a liberal world.", "Nov 2021", 345, 167),
        ("organic_critic", "Thoughtful AI Critic", "I disagree with some of Askell's approaches to Claude's constitution, particularly around moral relativism in edge cases. Worth reading the actual document though.", -0.5, 12000, 800, "AI safety researcher. PhD. Opinions are my own. Previously at DeepMind.", "Mar 2019", 234, 12),
    ]

    query = "Amanda Askell has no children"
    posts = []
    profiles = {}

    for handle, name, text, hours_ago, followers, following, bio, joined, likes, reposts in posts_data:
        ts = (now + timedelta(hours=hours_ago)).isoformat()
        posts.append({
            "handle": handle,
            "display_name": name,
            "text_snippet": text,
            "timestamp": ts,
            "followers": followers,
            "following": following,
            "bio": bio,
            "joined": joined,
            "likes": likes,
            "reposts": reposts,
            "url": f"https://x.com/{handle}/status/example",
        })
        profiles[handle] = {
            "handle": handle,
            "display_name": name,
            "bio": bio,
            "followers": followers,
            "following": following,
            "joined": joined,
        }

    return posts, profiles, query


# ── Wizard Mode ────────────────────────────────────────────────────────

async def wizard():
    """Interactive wizard for users who just want to run the thing."""
    banner()

    print(f"""  {C.BOLD}Welcome to Lysis Unraveler.{C.RESET}

  This tool detects coordinated campaigns on X (Twitter).
  It searches for a phrase you've seen repeated, collects
  every account that posted it, and runs statistical tests
  to determine if the pattern is organic or coordinated.

  {C.DIM}No API keys needed. No paid services. Just a browser.{C.RESET}
""")

    # Step 1: Choose mode
    step(1, "What would you like to do?")
    print(f"""
      {C.PURPLE}a){C.RESET} Search X for a phrase           {C.DIM}(opens browser, you log in){C.RESET}
      {C.PURPLE}b){C.RESET} Run demo with sample data       {C.DIM}(no login needed, test the tool){C.RESET}
      {C.PURPLE}c){C.RESET} Analyze existing JSON report     {C.DIM}(re-run analysis on saved data){C.RESET}
""")
    mode = ask("Choose a, b, or c", "a").lower()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    if mode == "b":
        await run_demo(script_dir)
        return

    if mode == "c":
        json_path = ask("Path to JSON report file")
        if not os.path.exists(json_path):
            error(f"File not found: {json_path}")
            return
        with open(json_path) as f:
            report = json.load(f)
        text_report = generate_report(report)
        print("\n" + text_report)
        if confirm("Open 3D visualization?"):
            open_visualization(script_dir, json_path, report)
        return

    # Mode A: Full search
    step(2, "What phrase are you seeing repeated?")
    info("Paste the exact text you've seen duplicated across accounts.")
    info("The tool will search X for this exact phrase in quotes.")
    print()
    query = ask("Search phrase")

    if not query:
        error("No phrase entered. Exiting.")
        return

    step(3, "How deep should we search?")
    info("More scrolls = more posts found, but takes longer.")
    info("10 scrolls ~ 1-2 minutes. 30 scrolls ~ 5-8 minutes.")
    print()
    max_scrolls = int(ask("Number of scrolls", "15"))

    step(4, "Where should we save the report?")
    default_output = os.path.join(script_dir, "coordination_report.json")
    output_path = ask("Output file", default_output)

    cookie_path = os.path.join(script_dir, ".x_cookies.json")

    # Summary before launch
    print(f"""
{C.CYAN}{'─' * 60}{C.RESET}
  {C.BOLD}Ready to launch:{C.RESET}

  Phrase:    "{C.YELLOW}{query}{C.RESET}"
  Scrolls:   {max_scrolls}
  Output:    {output_path}
{C.CYAN}{'─' * 60}{C.RESET}
""")

    if not confirm("Start the search?"):
        info("Cancelled.")
        return

    # Step 5: Check Playwright
    step(5, "Checking browser setup...")
    try:
        from playwright.async_api import async_playwright
        success("Playwright found.")
    except ImportError:
        error("Playwright not installed!")
        print(f"""
  {C.BOLD}Run these two commands to install it:{C.RESET}

    pip install playwright
    playwright install chromium

  {C.DIM}Then run this tool again.{C.RESET}
""")
        return

    # Step 6: Collect
    step(6, "Opening browser and searching X...")
    info("A browser window will open. Log into X if prompted.")
    info("Then come back to this terminal and press Enter.")
    print()

    posts, profiles = await collect_posts(
        query,
        max_scrolls=max_scrolls,
        headless=False,
        cookie_file=cookie_path,
    )

    if not posts:
        error("No posts found. Try a different phrase or check your X login.")
        return

    success(f"Collected {len(posts)} posts from {len(set(p.get('author_handle','') for p in posts))} accounts.")

    # Step 7: Analyze
    step(7, "Running coordination analysis...")
    info("Testing text similarity, temporal clustering, bio patterns, account profiles...")
    print()

    report = analyze_coordination(posts, profiles, query)

    # Show results
    text_report = generate_report(report)
    print("\n" + text_report)

    # Step 8: Save
    step(8, "Saving results...")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    success(f"JSON report  → {output_path}")

    text_path = output_path.replace(".json", "_report.txt")
    with open(text_path, "w") as f:
        f.write(text_report)
    success(f"Text report  → {text_path}")

    # Step 9: Visualization
    step(9, "3D Visualization")
    if confirm("Open the interactive 3D coordination web?"):
        open_visualization(script_dir, output_path, report)

    # Done
    print(f"""
{C.GREEN}{C.BOLD}{'═' * 60}
  DONE. Here's what was generated:
{'═' * 60}{C.RESET}

  {C.CYAN}JSON data:{C.RESET}   {output_path}
  {C.CYAN}Text report:{C.RESET} {text_path}

  {C.DIM}Share the report. Fork the tool. Sunlight is the best disinfectant.{C.RESET}
  {C.DIM}github.com/Cosmolalia/Lysis-Unraveler{C.RESET}
""")


async def run_demo(script_dir):
    """Run a full demo with synthetic data."""
    step(2, "Generating demo campaign data...")
    info("18 simulated accounts, realistic coordination patterns.")
    print()

    posts, profiles, query = generate_demo_report()
    success(f"Generated {len(posts)} posts from {len(profiles)} accounts.")

    step(3, "Running coordination analysis on demo data...")
    info("Same statistical tests as a live search.")
    print()

    report = analyze_coordination(posts, profiles, query)

    text_report = generate_report(report)
    print("\n" + text_report)

    # Save
    output_path = os.path.join(script_dir, "demo_report.json")
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    success(f"Demo report → {output_path}")

    text_path = output_path.replace(".json", "_report.txt")
    with open(text_path, "w") as f:
        f.write(text_report)
    success(f"Text report → {text_path}")

    step(4, "3D Visualization")
    if os.environ.get("LYSIS_NO_PROMPT"):
        # Non-interactive mode (for testing)
        open_visualization(script_dir, output_path, report)
    elif confirm("Open the interactive 3D coordination web?"):
        open_visualization(script_dir, output_path, report)

    cs = report.get("coordination_score", {})
    print(f"""
{C.GREEN}{C.BOLD}{'═' * 60}
  DEMO COMPLETE
{'═' * 60}{C.RESET}

  Coordination score: {C.RED}{C.BOLD}{cs.get('score', 0):.0%}{C.RESET} — {cs.get('verdict', '?')}

  {C.DIM}This was demo data. To search X for real:{C.RESET}
  {C.BOLD}python3 {os.path.basename(__file__)}{C.RESET}
  {C.DIM}Then choose option (a) and enter the phrase you're investigating.{C.RESET}
""")


def open_visualization(script_dir, json_path, report=None):
    """Create and open the 3D visualization."""
    viz_path = os.path.join(script_dir, "coordination_web.html")
    if not os.path.exists(viz_path):
        warn(f"coordination_web.html not found at {viz_path}")
        info("Download it from github.com/Cosmolalia/Lysis-Unraveler")
        return

    auto_viz = json_path.replace(".json", "_viz.html")
    create_autoload_viz(viz_path, json_path, auto_viz)
    success(f"3D visualization → {auto_viz}")

    import webbrowser
    webbrowser.open(f"file://{os.path.abspath(auto_viz)}")
    info("Opening in your browser...")


def create_autoload_viz(template_path, json_path, output_path):
    """Create a self-contained HTML viz that auto-loads the report data."""
    with open(template_path) as f:
        html = f.read()
    with open(json_path) as f:
        json_data = f.read()

    inject = f"""
<script>
// Auto-load report data
(function() {{
  const data = {json_data};
  window.addEventListener('load', () => {{
    setTimeout(() => loadReport(data), 500);
  }});
}})();
</script>
"""
    html = html.replace("</body>", inject + "</body>")

    with open(output_path, "w") as f:
        f.write(html)


# ── CLI Entrypoint ─────────────────────────────────────────────────────

async def main():
    """Main entrypoint — wizard mode if no args, CLI mode if args given."""
    if len(os.sys.argv) <= 1:
        # No arguments: run the wizard
        await wizard()
        return

    # Arguments given: run in CLI mode (for scripting/advanced users)
    parser = argparse.ArgumentParser(
        description="Detect coordinated inauthentic behavior on X",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Run with no arguments for interactive wizard mode.

Examples:
  python3 x_coordination_detector.py                           # wizard
  python3 x_coordination_detector.py "phrase to search"        # direct
  python3 x_coordination_detector.py "phrase" --max-scroll 25  # more posts
  python3 x_coordination_detector.py --demo                    # demo mode
        """
    )
    parser.add_argument("query", nargs="?", default=None, help="Phrase to search for on X")
    parser.add_argument("--demo", action="store_true", help="Run with demo data (no X login needed)")
    parser.add_argument("--max-scroll", type=int, default=15, help="Max scrolls (default: 15)")
    parser.add_argument("--headless", action="store_true", help="Run browser without window (needs saved cookies)")
    parser.add_argument("--output", "-o", default=None, help="Output JSON path")
    parser.add_argument("--cookies", default=None, help="Cookie file path")
    parser.add_argument("--no-viz", action="store_true", help="Skip 3D visualization")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    banner()

    if args.demo:
        os.environ["LYSIS_NO_PROMPT"] = "1"
        await run_demo(script_dir)
        return

    if not args.query:
        await wizard()
        return

    output_path = args.output or os.path.join(script_dir, "coordination_report.json")
    cookie_path = args.cookies or os.path.join(script_dir, ".x_cookies.json")

    step(1, f'Searching X for: "{args.query}"')
    info(f"Max scrolls: {args.max_scroll} | Output: {output_path}")

    posts, profiles = await collect_posts(
        args.query,
        max_scrolls=args.max_scroll,
        headless=args.headless,
        cookie_file=cookie_path,
    )

    if not posts:
        error("No posts found.")
        return

    success(f"Collected {len(posts)} posts.")

    step(2, "Analyzing...")
    report = analyze_coordination(posts, profiles, args.query)
    text_report = generate_report(report)
    print("\n" + text_report)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    success(f"JSON → {output_path}")

    text_path = output_path.replace(".json", "_report.txt")
    with open(text_path, "w") as f:
        f.write(text_report)
    success(f"Text → {text_path}")

    if not args.no_viz:
        open_visualization(script_dir, output_path, report)


if __name__ == "__main__":
    asyncio.run(main())
