# Recruiter–Candidate Call Analytics (NLP R&D)

*[Русская версия ниже / Russian version below](#рекрутер–кандидат-аналитика-звонков-nlp-rd)*

An exploratory NLP notebook that analyzes diarized, transcribed recruiter–candidate phone calls: it checks whether a required conversation checklist was covered, scores checklist compliance on a 0–10 scale, and explores speech-pattern signals (talk-time ratio, part-of-speech usage, emotion) as potential call-quality indicators.

## What it does

- Parses nested JSON call transcriptions into a flat, per-utterance dataset (recruiter vs. candidate).
- Classifies utterances into checklist topics (greeting, citizenship, age, health, salary, employment paperwork) using a Russian-language keyword/regex classifier.
- Computes a **0–10 compliance score** measuring how much of the required checklist was actually covered.
- Generalizes the above into a **reusable keyword-extraction engine**: feed it any free-text checklist, and it tokenizes, lemmatizes, stems, and matches it against a transcript to produce the same compliance score.
- Runs exploratory analysis on speech patterns: recruiter/candidate talk-time ratio, part-of-speech distribution, and a phik correlation matrix between speaker, text-based emotion, and audio-based emotion.
- Generates word clouds for the full corpus and per speaker role.

## Tech stack

Python 3, Google Colab · pandas, numpy · NLTK (tokenization, stopwords, WordNetLemmatizer, SnowballStemmer) · pymorphy3 · custom Russian Porter stemmer · re · phik · matplotlib, seaborn, plotly.express, wordcloud · tqdm · torch (reserved for future embedding-based work)

## Status

Research/prototype stage — a methodology exploration, not a production pipeline. All company-, candidate-, and vacancy-specific data has been generalized/omitted.

---

# Рекрутер–кандидат: аналитика звонков (NLP R&D)

Исследовательский NLP-ноутбук для анализа размеченных по спикерам транскрипций звонков «рекрутер—кандидат»: проверяет, покрыт ли обязательный чек-лист тем разговора, оценивает соответствие чек-листу по шкале 0–10 и исследует речевые паттерны (соотношение активности, части речи, эмоции) как потенциальные индикаторы качества звонка.

## Что делает проект

- Разбирает вложенный JSON транскрипции звонков в плоский датасет реплик (рекрутер / кандидат).
- Классифицирует реплики по темам чек-листа (приветствие, гражданство, возраст, здоровье, зарплата, оформление) с помощью классификатора на ключевых словах/регулярных выражениях для русского языка.
- Рассчитывает **метрику соответствия 0–10**, показывающую, насколько полно был покрыт требуемый чек-лист.
- Обобщает подход в **переиспользуемый механизм извлечения ключевых слов**: на вход подаётся произвольный текст чек-листа — он токенизируется, лемматизируется, приводится к основе и сопоставляется с транскрипцией для получения той же метрики.
- Проводит разведочный анализ речевых паттернов: соотношение активности рекрутера и кандидата, распределение частей речи, матрицу корреляции phik между спикером, эмоцией по тексту и эмоцией по аудио.
- Строит облака слов для всего корпуса и отдельно по ролям.

## Технологический стек

Python 3, Google Colab · pandas, numpy · NLTK (токенизация, стоп-слова, WordNetLemmatizer, SnowballStemmer) · pymorphy3 · собственный стеммер Портера для русского языка · re · phik · matplotlib, seaborn, plotly.express, wordcloud · tqdm · torch (задел под работу с эмбеддингами)

## Статус

Исследовательский прототип — методологический эксперимент, а не production-пайплайн. Все данные, специфичные для компании, кандидатов и вакансий, обобщены/исключены.
