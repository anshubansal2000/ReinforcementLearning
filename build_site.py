"""Generate the static Vercel website from the publication article.

Run after editing blog/reinforcement-learning.md or regenerating its images.
Vercel serves the committed website/ output without installing Python dependencies.
"""
from pathlib import Path
import html
import re
import shutil
import mistune

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "website"
OUT.mkdir(exist_ok=True)
ARTICLE = ROOT / "blog/reinforcement-learning.md"
text = ARTICLE.read_text(encoding="utf-8-sig")
title = text.splitlines()[0].removeprefix("# ")
subtitle = text.splitlines()[2].strip("*")
body = "\n".join(text.splitlines()[3:]).strip()
body = re.sub(r"^\*\*(.+?)\*\*$", r"## \1", body, flags=re.M)
body = body.replace("(images/", "(assets/").replace("(../results/convergence.png)", "(assets/convergence.png)")
rendered = mistune.create_markdown(plugins=["table"])(body)
headings = []
def heading(match):
    label = match.group(1)
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    headings.append((slug, label))
    return f'<h2 id="{slug}">{label}</h2>'
rendered = re.sub(r"<h2>(.*?)</h2>", heading, rendered)
rendered = rendered.replace("<table>", '<div class="table-scroll" tabindex="0" role="region" aria-label="Experiment comparison"><table>').replace("</table>", "</table></div>")
def image(match):
    src, alt = match.groups()
    return f'<figure><a href="{src}" aria-label="Open full-size graphic: {alt}"><img src="{src}" alt="{alt}" loading="lazy" decoding="async"></a></figure>'
rendered = re.sub(r'<p><img src="([^"]+)" alt="([^"]*)"\s*/?></p>', image, rendered)
toc = "".join(f'<li><a href="#{slug}">{label}</a></li>' for slug,label in headings)
repo = "https://github.com/anshubansal2000/ReinforcementLearning"
colab = "https://colab.research.google.com/github/anshubansal2000/ReinforcementLearning/blob/main/mars_rover_assignment.ipynb"
description = "We built two Gymnasium worlds to compare DP, Monte Carlo, and TD learning—and found why 20,000 episodes can still leave blind spots."
document = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} | Anshu Bansal</title>
<meta name="description" content="{html.escape(description)}">
<meta name="theme-color" content="#132e3b">
<meta property="og:type" content="article"><meta property="og:title" content="{html.escape(title)}">
<meta property="og:description" content="{html.escape(description)}">
<link rel="canonical" href="https://reinforcementlearning.vercel.app/">
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<link rel="stylesheet" href="/styles.css">
</head><body>
<a class="skip" href="#article">Skip to article</a>
<header class="topbar"><a class="brand" href="/" aria-label="RL Field Notes home"><span class="mark">RL</span><span>FIELD NOTES</span></a>
<a class="source-link" href="{repo}">Code on GitHub <span aria-hidden="true">↗</span></a></header>
<main>
<section class="hero" aria-labelledby="title">
<div class="eyebrow"><span class="dot"></span> REINFORCEMENT LEARNING / EXPERIMENT 01</div>
<h1 id="title">We built a Mars rover to understand <em>how RL actually learns.</em></h1>
<p class="dek">{html.escape(subtitle)}</p>
<div class="byline"><span class="avatar" aria-hidden="true">AB</span><div><strong>Anshu Bansal</strong><span>September 14, 2026 · 13 min read</span></div></div>
<div class="actions"><a class="button primary" href="{colab}">Run the experiment <span aria-hidden="true">↗</span></a><a class="button secondary" href="{repo}/blob/main/mars_rover_assignment.ipynb">Read the notebook <span aria-hidden="true">↗</span></a></div>
</section>
<div class="reading-layout">
<aside class="contents" aria-label="Article navigation"><details open><summary>IN THIS ARTICLE</summary><ol>{toc}</ol></details></aside>
<article id="article">{rendered}</article>
</div>
</main>
<footer><div><strong>RL FIELD NOTES</strong><p>Experiments, explanations, and evidence.</p></div><a href="{repo}">Explore the code on GitHub ↗</a></footer>
</body></html>'''
(OUT / "index.html").write_text(document, encoding="utf-8")
assets = OUT / "assets"
assets.mkdir(exist_ok=True)
for path in (ROOT / "blog/images").glob("*.png"):
    shutil.copy2(path, assets / path.name)
shutil.copy2(ROOT / "results/convergence.png", assets / "convergence.png")
(OUT / "robots.txt").write_text("User-agent: *\nAllow: /\nSitemap: https://reinforcementlearning.vercel.app/sitemap.xml\n")
(OUT / "sitemap.xml").write_text('<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><url><loc>https://reinforcementlearning.vercel.app/</loc></url></urlset>')
assert len(headings) == 13, len(headings)
assert len(re.findall(r'<figure>', document)) == 6
for path in re.findall(r'<img src="([^"]+)"', document):
    assert (OUT / path).is_file(), path
print(f"Built article with {len(headings)} sections and six figures")
