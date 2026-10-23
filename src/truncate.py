"""Choose which 200 words of each advertisement the models read.

The text is cut on words, not tokens, so every method reads the same text.
v1 read the first 200 words. v2 keeps the opening and then skips the employer
introduction to the job section, which scored 0.826 against 0.755 with TF-IDF
on the validation split.
"""

WORD_BUDGET = 200
INPUT_STRATEGY = "opening_job_wide"

# Headings where the job description starts, found in the training ads.
JOB_MARKERS = ["arbetsuppgifter", "om tjänsten", "om jobbet", "om rollen",
               "i rollen", "arbetet innebär", "dina uppgifter",
               "ansvarsområden", "om befattningen", "arbetsbeskrivning",
               "ditt uppdrag", "om uppdraget", "uppdragsbeskrivning",
               "jobbeskrivning", "information om uppdraget"]

# The opening often names the job, so it is kept.
OPENING_WORDS = 30


def first_words(text, n_words):
    """Return the first n_words words of the text."""
    return " ".join(text.split()[:n_words])


def skip_intro(text):
    """Return the text from the first job heading, or all of it if there is none."""
    low = text.lower()
    starts = [low.find(marker) for marker in JOB_MARKERS if marker in low]

    if not starts:
        return text

    return text[min(starts):]


def opening_and_job(text, budget):
    """Return the opening words followed by the job section.

    Args:
        text: The full advertisement text.
        budget: How many words to keep in total.

    Returns:
        The first OPENING_WORDS words and then the job section, or simply the
        first budget words if there is no intro to skip.
    """
    words = text.split()
    job = skip_intro(text).split()

    if len(words) - len(job) <= OPENING_WORDS:
        return " ".join(words[:budget])

    return " ".join(words[:OPENING_WORDS] + job[:budget - OPENING_WORDS])


def truncate_text(text, budget, strategy):
    """Cut a text down to the budget.

    Args:
        text: The full advertisement text.
        budget: How many words to keep.
        strategy: "head" (v1), "opening_job_wide" (v2) or "none".

    Returns:
        The words the model reads.
    """
    if strategy == "none":
        return text
    if strategy == "head":
        return first_words(text, budget)
    if strategy == "opening_job_wide":
        return opening_and_job(text, budget)

    raise ValueError(f"Unknown truncation strategy: {strategy}")
