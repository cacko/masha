from textacy.preprocessing import remove, replace, pipeline
from markdown import markdown


def clean_markdown(s: str):
    html = markdown(s)
    clean_pipeline = pipeline.make_pipeline(
        remove.accents,
        remove.brackets,
        remove.html_tags,
        replace.emojis,
        replace.hashtags,
    )
    return clean_pipeline(html)
