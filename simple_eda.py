#!/usr/bin/env python
# lendingclub_eda.py
# ---------------------------------------------------------------
# One-file exploratory tool for Lending Club (or any tabular data)
# ---------------------------------------------------------------

import os, json, textwrap, inspect, functools
from collections import defaultdict
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# ===============================================================
# 0. CONFIG  ––––––––––––––––––––––––––––––––––––––––––––––––––––
# ===============================================================

# ⚠️ point this at your Lending Club CSV (or any CSV with header row)
CSV_PATH = r"C:\Users\hendr\Downloads\LendingClubData_31July2024\credit_card_2015_2016.csv"

# parse only the header (schema) first – real data loaded later on demand
LOAD_FULL_DATA = True          # set False if you have a huge file

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")     # put it in .env or hard-code
GPT_MODEL      = "gpt-3.5-turbo"

# ===============================================================
# 1.  Schema-only EDA helper  –––––––––––––––––––––––––––––––––––
# ===============================================================
class SchemaEDA:
    _num  = {"int", "float", "double", "decimal"}
    _cat  = {"category", "object"}
    _date = {"date", "datetime", "time"}
    _bool = {"bool", "boolean"}

    def __init__(self, header_df: pd.DataFrame):
        """
        header_df : DataFrame with columns ['name','dtype']
                    dtype may be 'unknown'
        """
        self.df    = header_df
        self.types = self._bucketize()

    def _bucketize(self):
        out = defaultdict(list)
        for _, row in self.df.iterrows():
            n, t = row["name"], str(row["dtype"]).lower()
            if any(k in t for k in self._num):
                out["numeric"].append(n)
            elif any(k in t for k in self._date):
                out["date"].append(n)
            elif any(k in t for k in self._bool):
                out["boolean"].append(n)
            elif any(k in t for k in self._cat):
                out["categorical"].append(n)
            else:
                out["text"].append(n)
        return out

    def plan(self) -> str:
        s = self.types
        return textwrap.dedent(f"""
        ------------------------------------------------------------
        ️📋  Exploratory-Data-Analysis Checklist
        ------------------------------------------------------------
        • Numeric cols   : {len(s['numeric'])}
        • Categorical    : {len(s['categorical'])}
        • Text           : {len(s['text'])}
        • Boolean        : {len(s['boolean'])}
        • Date/Datetime  : {len(s['date'])}

        Suggested first steps
        ---------------------
        1. missing_values
        2. describe_numeric
        3. value_counts <column>
        4. correlation_heatmap
        5. class_balance (if a target column exists)
        ------------------------------------------------------------
        Ask in plain English and GPT will call the right command.
        Type 'help' to list available actions or 'quit' to exit.
        """)

# ===============================================================
# 2.  COMMAND REGISTRY  –––––––––––––––––––––––––––––––––––––––––
# ===============================================================
COMMANDS = {}

def cmd(name, desc):
    def deco(f):
        COMMANDS[name] = (f, desc)
        return functools.wraps(f)(f)
    return deco

@cmd("show_head", "Display the first 10 rows of the data")
def show_head(df):
    print(df.head(10))

@cmd("describe_numeric", "Summary statistics of numeric columns")
def describe_numeric(df):
    print(df.select_dtypes("number").describe().T)

@cmd("missing_values", "Percentage of missing values per column")
def missing_values(df):
    mv = df.isna().mean().mul(100).round(2)
    print((mv[mv>0]).sort_values(ascending=False).to_string())

@cmd("value_counts", "Top value counts for a given column (usage: value_counts <col>)")
def value_counts(df, col=None):
    if not col:
        print("⚠️  Usage: value_counts <column_name>"); return
    if col not in df.columns:
        print(f"Column '{col}' not found"); return
    print(df[col].value_counts(dropna=False).head(20))

@cmd("correlation_heatmap", "Correlation heat-map of numeric features")
def correlation_heatmap(df):
    import seaborn as sns, matplotlib.pyplot as plt
    sns.heatmap(df.select_dtypes("number").corr(), cmap="coolwarm")
    plt.show()

@cmd("class_balance", "Bar plot of the target column distribution (usage: class_balance <target>)")
def class_balance(df, target=None):
    if not target:
        print("⚠️ Usage: class_balance <target_column>"); return
    import matplotlib.pyplot as plt
    df[target].value_counts(dropna=False).plot.bar()
    plt.title("Class Balance")
    plt.show()

# helper
def list_commands():
    for k,(f,d) in COMMANDS.items():
        sig = inspect.signature(f)
        args = " ".join([p.name for p in sig.parameters.values()][1:])  # skip df arg
        print(f"* {k} {args:<15} – {d}")

# ===============================================================
# 3.  GPT dispatcher  –––––––––––––––––––––––––––––––––––––––––––
# ===============================================================
def dispatch(df, user_text):
    """
    Map natural language to one registered command & execute.
    """
    if user_text.lower() in {"help","h","commands"}:
        list_commands(); return

    # manual override: allow user to type command directly
    tok = user_text.split()
    if tok[0] in COMMANDS:
        _execute(df, tok[0], tok[1:]); return

    # ---------- use GPT -----------------------------------------
    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY missing.  "
              "Type an exact command name or set the key."); return
    import openai
    openai.api_key = OPENAI_API_KEY

    choices = "\n".join([f"- {k}: {d}" for k,(_,d) in COMMANDS.items()])
    sys_msg = (
        "You are an assistant that maps a user's request onto exactly ONE "
        "command name from the list below. Reply ONLY with that command "
        "name and, if required, its argument separated by space. Nothing else.\n\n"
        f"Commands:\n{choices}"
    )

    r = openai.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role":"system","content":sys_msg},
                  {"role":"user",  "content":user_text}],
        max_tokens=8,
        temperature=0
    )
    cmd_line = r.choices[0].message.content.strip()
    _execute(df, *cmd_line.split(maxsplit=1))

def _execute(df, name, arg_str=""):
    if name not in COMMANDS:
        print(f"🤖 replied with unknown command: '{name}'"); return
    func = COMMANDS[name][0]
    args = arg_str.split() if arg_str else []
    print(f"\n▶ {name} {' '.join(args)}\n" + "-"*60)
    func(df, *args)

# ===============================================================
# 4.  RUN
# ===============================================================
if __name__ == "__main__":
    # 4-a) build schema DataFrame
    hdr = pd.read_csv(CSV_PATH, nrows=0).columns
    schema_df = pd.DataFrame({"name": hdr, "dtype": "unknown"})
    eda = SchemaEDA(schema_df)

    print(eda.plan())

    # 4-b) optionally load the full data
    df = pd.DataFrame()
    if LOAD_FULL_DATA:
        print("📥  Loading full dataset …")
        df = pd.read_csv(CSV_PATH)
        print("Loaded rows:", len(df))

    # 4-c) interactive loop
    while True:
        prompt = input("\n💬 Ask in English> ").strip()
        if prompt.lower() in {"quit","exit","q"}:
            break
        dispatch(df, prompt)
