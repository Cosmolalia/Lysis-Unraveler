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
    from playwright.async_api import async_playwright

    posts = []
    seen_ids = set()

    async with async_playwright() as p:
        # Launch browser (visible so user can log in)
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

        # Load cookies if available
        if cookie_file and os.path.exists(cookie_file):
            with open(cookie_file) as f:
                cookies = json.load(f)
            await context.add_cookies(cookies)
            print(f"  [cookies] Loaded from {cookie_file}")

        page = await context.new_page()

        # Navigate to X
        print("  [browser] Opening X...")
        await page.goto("https://x.com", wait_until="domcontentloaded")
        await page.wait_for_timeout(3000)

        # Check if logged in
        logged_in = False
        try:
            # Look for search box or compose button (signs of logged-in state)
            search = await page.query_selector('a[href="/explore"]')
            if search:
                logged_in = True
        except:
            pass

        if not logged_in:
            print("\n" + "=" * 60)
            print("  LOG IN TO X in the browser window.")
            print("  Then come back here and press ENTER.")
            print("=" * 60)
            input("\n  Press ENTER when logged in > ")
            await page.wait_for_timeout(2000)

            # Save cookies for next time
            if cookie_file:
                cookies = await context.cookies()
                with open(cookie_file, "w") as f:
                    json.dump(cookies, f)
                print(f"  [cookies] Saved to {cookie_file}")

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

        await browser.close()

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
    report = {
        "query": query,
        "collection_time": datetime.now(timezone.utc).isoformat(),
        "total_posts": len(posts),
        "unique_authors": len(set(p.get("author_handle", "") for p in posts)),
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


# ── Main ───────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(
        description="Detect coordinated inauthentic behavior on X",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 x_coordination_detector.py "Amanda Askell has no children"
  python3 x_coordination_detector.py "exact phrase to search" --max-scroll 25
  python3 x_coordination_detector.py "phrase" --headless  # no browser window
        """
    )
    parser.add_argument("query", help="The phrase to search for on X")
    parser.add_argument("--max-scroll", type=int, default=15, help="Max scrolls in search results (default: 15)")
    parser.add_argument("--headless", action="store_true", help="Run browser without visible window (needs saved cookies)")
    parser.add_argument("--output", "-o", default=None, help="Output JSON file (default: data/coordination_report.json)")
    parser.add_argument("--cookies", default=None, help="Cookie file path for persistent login")
    parser.add_argument("--no-viz", action="store_true", help="Skip opening the 3D visualization")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = args.output or os.path.join(script_dir, "coordination_report.json")
    cookie_path = args.cookies or os.path.join(script_dir, ".x_cookies.json")

    print("=" * 70)
    print("  X COORDINATION DETECTOR")
    print("  Searches for coordinated inauthentic behavior")
    print("=" * 70)
    print(f"  Query: \"{args.query}\"")
    print(f"  Max scrolls: {args.max_scroll}")
    print(f"  Output: {output_path}")
    print()

    # Collect
    posts, profiles = await collect_posts(
        args.query,
        max_scrolls=args.max_scroll,
        headless=args.headless,
        cookie_file=cookie_path,
    )

    if not posts:
        print("\n  NO POSTS FOUND. Try a different search query.")
        return

    # Analyze
    print("\n  [analysis] Running coordination detection...")
    report = analyze_coordination(posts, profiles, args.query)

    # Report
    text_report = generate_report(report)
    print("\n" + text_report)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  [save] Full report → {output_path}")

    # Also save text report
    text_path = output_path.replace(".json", "_report.txt")
    with open(text_path, "w") as f:
        f.write(text_report)
    print(f"  [save] Text report → {text_path}")

    # Open visualization
    if not args.no_viz:
        viz_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "coordination_web.html")
        if os.path.exists(viz_path):
            import webbrowser
            # The HTML loads JSON via drag-drop, so also create an auto-load version
            auto_viz = output_path.replace(".json", "_viz.html")
            create_autoload_viz(viz_path, output_path, auto_viz)
            print(f"  [viz] Opening 3D coordination web → {auto_viz}")
            webbrowser.open(f"file://{os.path.abspath(auto_viz)}")
        else:
            print(f"  [viz] coordination_web.html not found at {viz_path}")


def create_autoload_viz(template_path, json_path, output_path):
    """Create a self-contained HTML viz that auto-loads the report data."""
    with open(template_path) as f:
        html = f.read()
    with open(json_path) as f:
        json_data = f.read()

    # Inject auto-load script before closing </body>
    inject = f"""
<script>
// Auto-load report data
(function() {{
  const data = {json_data};
  // Wait for page load, then auto-load
  window.addEventListener('load', () => {{
    setTimeout(() => loadReport(data), 500);
  }});
}})();
</script>
"""
    html = html.replace("</body>", inject + "</body>")

    with open(output_path, "w") as f:
        f.write(html)


if __name__ == "__main__":
    asyncio.run(main())
