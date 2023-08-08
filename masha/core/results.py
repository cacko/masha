from humanfriendly.tables import (
    format_robust_table,
    format_pretty_table
)
from stringcase import titlecase
from coreface import FaceTestResult
from masha.core.term import ccze, TermColor
from itertools import groupby


def shorten(text: str, size: int, placeholder: str = "..", extraSize=0):
    if not isinstance(text, str):
        return text
    size += max([0, extraSize])
    if len(text) <= size:
        return text.strip()
    return f"{text.strip()[:size-len(placeholder)].strip()}{placeholder}"


def robust(results: list[FaceTestResult]):
    cols = [
        "Find",
        "Verify",
        "File",
        "Targes",
        "Matched",
        "Results"
    ]

    rows = []

    for res in results:
        rows.append([
            res.find_config.description,
            res.verify_config.description,
            res.img.name,
            ",".join(map(titlecase, sorted(res.targets))),
            ",".join(map(titlecase, sorted(res.matches))),
            " / ".join([
                f"valid: {res.valid_count}",
                f"to match: {res.target_count}",
                f"found: {res.match_count}"
            ])
        ])

    return format_robust_table(rows, column_names=cols)


def pretty(results: list[FaceTestResult], cases: list[str]):
    columns = ["Config", *[shorten(x, 15) for x in cases]]
    rows = []
    for group, rsts in groupby(results, lambda x: '.'.join(
            [x.find_config.name, x.verify_config.name])):
        row = [group]
        for res in rsts:
            vcb, tcb, mcb = None, None, None
            vc, tc, mc = None, None, None
            if res.valid_count == res.target_count:
                vc = TermColor.GREEN
                vcb = True

            row.append(
                "/".join([
                    ccze(f"{res.valid_count}", color=vc, bright=vcb),  # type: ignore
                    ccze(f"{res.target_count}", color=tc, bright=tcb),  # type: ignore
                    ccze(f"{res.match_count}", color=mc, bright=mcb)  # type: ignore
                ])
            )
        rows.append(row)
    return format_pretty_table(rows, columns)
