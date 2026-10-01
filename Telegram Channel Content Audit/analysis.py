"""
Telegram Channel Content Audit — analysis script
==================================================

Reproduces the full quantitative analysis behind WAY2WEI_TG_content_analysis.md.

Input: a CSV with columns [id, date, text, url] — a standard export shape
produced by common Telegram-channel scraping tools (e.g. a Colab Telethon
export). Raw channel data is NOT included in this repo (the post texts
belong to the channel owner) — provide your own export as `posts.csv` in
the same directory, or point DATA_PATH to your file, to reproduce the run.

Usage:
    python analysis.py posts.csv

Dependencies: pandas (standard library `re`, `collections` otherwise)
"""

import sys
import re
from collections import Counter

import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)


# ---------------------------------------------------------------------------
# 1. Load & basic prep
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.to_period("M")
    df["dow"] = df["date"].dt.day_name()
    df["hour"] = df["date"].dt.hour
    df["len"] = df["text"].str.len()
    return df


# ---------------------------------------------------------------------------
# 2. Cadence: volume, gaps, weekday/hour distribution, post length
# ---------------------------------------------------------------------------

def cadence_report(df: pd.DataFrame) -> None:
    print("\n=== Posts per year ===")
    print(df.groupby("year").size())

    print("\n=== Posts per month, last 24 months ===")
    print(df.groupby("month").size().tail(24))

    print("\n=== Posts by weekday ===")
    print(df["dow"].value_counts())

    print("\n=== Posts by hour of day ===")
    print(df["hour"].value_counts().sort_index())

    print("\n=== Post length (characters) ===")
    print(df["len"].describe())
    print("\nAverage length by year:")
    print(df.groupby("year")["len"].mean().round(0))


# ---------------------------------------------------------------------------
# 3. Content categorization (rule-based, transparent & auditable)
# ---------------------------------------------------------------------------

CATEGORY_RULES = [
    ("Holiday/greeting",
     r"поздравля|с праздником|с новым годом|8 марта|день знаний|день победы|международный женский"),
    ("Event/company participation",
     r"конференци|форум|митап|выступил[аи]|представител[ья] нашей команды|вебинар"),
    ("Client case study",
     r"кейс|результат.{0,20}клиент|hrd .{0,30}(?:рассказыва|делится)"),
    ("Job opening",
     r"вакансия|ищем в команду|присоединяйся к нам|открыта позиция"),
    ("Contest/giveaway",
     r"конкурс|розыгрыш|разыгрываем|приз получит"),
    ("Product/CTA",
     r"way2wei|тэи[- ]|подписка|купить|заказать|оставьте заявку|попробовать бесплатно|записаться"),
    ("Research/science",
     r"исследовани[ея]|учены[ех]|по данным науки|научн"),
    ("Entertainment rubric",
     r"кино на выходн|детектив"),
]


def classify_post(text: str) -> str:
    """Assigns the first matching category in priority order; defaults to
    'Educational/conceptual' if nothing matches. Priority order matters —
    e.g. a post mentioning both a conference AND the product name is counted
    as an event, not a CTA, because events are checked first."""
    tl = text.lower()
    for category, pattern in CATEGORY_RULES:
        if re.search(pattern, tl):
            return category
    return "Educational/conceptual"


def category_report(df: pd.DataFrame) -> pd.DataFrame:
    df["category"] = df["text"].apply(classify_post)

    print("\n=== Content mix (overall) ===")
    print(df["category"].value_counts())

    print("\n=== Content mix by year (cross-tab) ===")
    pivot = pd.crosstab(df["year"], df["category"])
    print(pivot.to_string())

    return df


# ---------------------------------------------------------------------------
# 4. Keyword-frequency comparison across periods (vocabulary shift)
# ---------------------------------------------------------------------------

STOPWORDS_RU = set(
    "и в во не что он на я с со как а то все она так его но да ты к у же вы "
    "за бы по только ее мне было вот от меня еще нет о из ему теперь когда "
    "даже ну вдруг ли если уже или ни быть был него до вас нибудь опять уж "
    "вам сказал ведь там потом себя ничего ей может они тут где есть надо "
    "ней для мы тебя их чем была сам чтоб без будто чего раз тоже себе под "
    "будет ж тогда кто этот того потому этого какой совсем ним здесь этом "
    "один почти мой тем чтобы нее сейчас были куда зачем всех никогда можно "
    "при наконец два об другой хоть после над больше тот через эти нас про "
    "всего них какая много разве три эту моя впрочем хорошо свою этой перед "
    "иногда лучше чуть том нельзя такой им более всегда конечно всю между "
    "это эта эти которые которая который которых которого которым каждый "
    "каждого каждой".split()
)


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[а-яё]{4,}", text.lower())
    return [w for w in words if w not in STOPWORDS_RU]


def vocabulary_shift_report(df: pd.DataFrame, early_until: int, recent_from: int, top_n: int = 25) -> None:
    df["tokens"] = df["text"].apply(tokenize)

    recent_tokens = df.loc[df["year"] >= recent_from, "tokens"].sum()
    early_tokens = df.loc[df["year"] < early_until, "tokens"].sum()

    print(f"\n=== Top {top_n} words, {recent_from}-present ===")
    for word, count in Counter(recent_tokens).most_common(top_n):
        print(word, count)

    print(f"\n=== Top {top_n} words, before {early_until} ===")
    for word, count in Counter(early_tokens).most_common(top_n):
        print(word, count)


# ---------------------------------------------------------------------------
# 5. Topic / industry coverage mapping (customize for your own niche list)
# ---------------------------------------------------------------------------

TOPIC_RULES = {
    "Burnout": r"выгора",
    "Attrition/turnover": r"текучест",
    "Hiring/recruiting": r"найм|рекрут|собеседован|кандидат",
    "Onboarding/adaptation": r"адаптац|онбординг",
    "Leadership/management": r"руководител|лидер",
    "Team": r"команд",
    "Product/platform": r"way2wei|тэи|платформ",
    "Client case": r"кейс",
    "Events/webinars": r"конференци|вебинар|форум|митап|встреч",
    "ROI/business value": r"прибыл|выручк|roi|бюджет|стоимост",
}

INDUSTRY_RULES = {
    "IT": r"\bit\b|айти|цифровые технологии|разработчик|программист",
    "Hospitality": r"ресторан|гостинич|отел(?:ь)?|horeca",
    "Manufacturing/logistics": r"производств|логистик|завод",
    "Finance/banking": r"банк|финанс",
    "Retail": r"ритейл|розниц|магазин",
    "Consulting/agencies": r"консалтинг|агентств",
    "EdTech/education": r"edtech|образовательн|школ|университет|вуз",
    "Startups": r"стартап",
    "Medicine": r"медицин|клиник|больниц|врач",
    "Call center/customer service": r"колл-центр|call.?center|клиентск(?:ий|ого) сервис",
}


def coverage_report(df: pd.DataFrame, rules: dict, title: str) -> None:
    texts = df["text"].str.lower()
    print(f"\n=== {title} ===")
    for name, pattern in rules.items():
        count = texts.str.contains(pattern, regex=True, na=False).sum()
        print(f"{name}: {count}")


# ---------------------------------------------------------------------------
# 6. Format audit: hashtags, external links, recurring rubrics
# ---------------------------------------------------------------------------

def format_audit(df: pd.DataFrame) -> None:
    hashtags = df["text"].str.findall(r"#\w+")
    counter = Counter()
    for tags in hashtags:
        counter.update(tags)

    print("\n=== Top 20 hashtags ===")
    for tag, count in counter.most_common(20):
        print(tag, count)

    external_links = df["text"].str.contains(r"https?://(?!t\.me)", regex=True, na=False)
    print(f"\nPosts with external (non t.me) links: {external_links.sum()}")

    question_openers = df["text"].apply(lambda t: "?" in t.split("\n")[0])
    print(f"Posts opening with a question: {question_openers.sum()}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(path: str) -> None:
    df = load_data(path)

    cadence_report(df)
    df = category_report(df)
    vocabulary_shift_report(df, early_until=2024, recent_from=2025)
    coverage_report(df, TOPIC_RULES, "Topic coverage")
    coverage_report(df, INDUSTRY_RULES, "Industry coverage")
    format_audit(df)


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "posts.csv"
    main(data_path)
