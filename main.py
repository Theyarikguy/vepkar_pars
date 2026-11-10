# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import zst
import pandas as pd
import pandas as pd
from lxml import etree
from pathlib import Path


def read_csv_zst(path: str | Path, **kw) -> pd.DataFrame:
    """Читает CSV.Zstd с разумными для этого корпуса дефолтами."""
    return pd.read_csv(
        path,
        compression="zstd",
        dtype=str,
        keep_default_na=False,
        **kw,
    )


def xml_sentence_to_plain(xml_str: str) -> str:
    """Достаёт текст предложения из <s>...</s>, сохраняя пунктуацию и регистр."""
    if not xml_str.strip():
        return ""
    try:
        root = etree.fromstring(f"<root>{xml_str}</root>")
    except etree.XMLSyntaxError:
        return ""
    return "".join(root.itertext())


def build_full_texts(lang: str, base_dir: str | Path = ".") -> pd.DataFrame:
    """Возвращает DataFrame с колонками text_id и full_text."""
    base = Path(base_dir)

    sentences = read_csv_zst(base / f"sentences_{lang}.csv.zst")

    sentences["sentence_id_int"] = sentences["sentence_id"].astype(int)
    sentences["plain"] = sentences["sentence_xml"].apply(xml_sentence_to_plain)

    full = (
        sentences
        .sort_values(["text_id", "sentence_id_int"])
        .groupby("text_id", as_index=False)
        .agg(full_text=("plain", " ".join))
    )

    # Только id + текст; пустые тексты -> "", чтобы не было NaN
    full["full_text"] = full["full_text"].fillna("")
    return full[["text_id", "full_text"]]


if __name__ == "__main__":
    full_texts_olo = build_full_texts("olo", base_dir="")
    full_texts_olo.to_csv(
        "texts_olo_full.csv",
        index=False,
        encoding="utf-8",
    )
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
