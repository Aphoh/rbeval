from pathlib import Path
from polars import DataFrame, read_ndjson
import polars as pl
import altair as alt


def score_eval(eval):
    res = None
    for rating in ["C", "I"]:
        for u in range(1, 6):
            grade = f"[[{rating}{u}]]"
            if grade in eval:
                res = grade
    assert res is not None, f"no grade found for {eval}"
    return res


def load_data():
    for file in Path(".").glob("all_*eval.jsonl"):
        df = read_ndjson(file)
        df = df.with_columns(
            pl.col("eval")
            .str.extract_all(r"\[\[(C|c|i|I)[1-5]\]\]")
            .list.unique()
            .alias("scores")
        )
        n_scores = df.get_column("scores").list.len()
        df = df.filter(n_scores == 1)
        df = df.with_columns(df.get_column("scores").list.first().alias("score")).drop(
            "scores"
        )
        yield (file.stem, df)


def plot_score_dist(name: str, df: DataFrame) -> alt.Chart:
    value_counts = df.get_column("score").value_counts().to_dict()
    values = value_counts["score"]
    counts = value_counts["count"]
    vc = {v: c for v, c in zip(values, counts)}
    xs = [f"[[I{u}]]" for u in range(5, 0, -1)] + [f"[[C{u}]]" for u in range(1, 6)]
    ys = [vc.get(x, 0) for x in xs]
    data = alt.Data(values=[{"Score": x, "Count": y} for x, y in zip(xs, ys)])
    return (
        alt.Chart(data)
        .mark_bar()
        .encode(x=alt.X("Score:N", sort=xs), y="Count:Q")
        .properties(title=name)
    )
