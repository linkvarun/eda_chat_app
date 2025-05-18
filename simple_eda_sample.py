#!/usr/bin/env python
# lendingclub_eda.py
# ---------------------------------------------------------------
# One-file exploratory tool for Lending Club (or any tabular data)
# ---------------------------------------------------------------

import os, json, textwrap, inspect, functools, textwrap
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np
from dotenv import load_dotenv
import sqlalchemy
from sklearn.tree import DecisionTreeClassifier, export_text
# import openai
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================================================
# 0. CONFIG  ––––––––––––––––––––––––––––––––––––––––––––––––––––
# ===============================================================

# ⚠️ point this at your Lending Club CSV (or any CSV with header row)
CSV_PATH = r"data.csv"

# parse only the header (schema) first – real data loaded later on demand
LOAD_FULL_DATA = True          # set False if you have a huge file

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")     # put it in .env or hard-code
GPT_MODEL = "gemini-2.0-flash"

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
        6. text_to_sql <query>
        7. simulate_decision <col> <+/-value>
        8. explore_tree <target>
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

@cmd("show_head", "Display the first 5 rows of the data")
def show_head(df):
    print(df.head())

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
    import matplotlib.pyplot as plt
    import seaborn as sns
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

# Text-to-SQL
def clean_sql_string(sql):
    # Remove code fences and extra backticks
    sql = sql.replace("```sql", "").replace("```", "").strip()
    return sql

@cmd("text_to_sql", "Text to SQL query execution (usage: text_to_sql <query in plain English>)")
def text_to_sql(df, *query_parts):
    import pandasql
    if not check_data_loaded(df):
        return

    user_query = " ".join(query_parts)
    if not user_query:
        print("Usage: text_to_sql <natural language query>")
        return

    # No API key fallback
    if not OPENAI_API_KEY:
        print("⚠️  No API key set. Type a raw SQL query instead like:")
        print("text_to_sql SELECT col1, SUM(col2) FROM data WHERE col3 = 'ABC'")
        return

    from openai import OpenAI
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    # Prepare system prompt
    schema_info = ", ".join(df.columns)
    sys_msg = (
        f"You are a SQL assistant. Convert the user's request into a valid SQL query "
        f"against a table data with these columns: {schema_info}. "
        f"Always use data as the table name. Only reply with the SQL query string."
    )

    try:
        # Call GPT to generate the SQL
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "system", "content": sys_msg},
                      {"role": "user", "content": user_query}],
            max_tokens=200,
            temperature=0
        )
        raw_sql = response.choices[0].message.content.strip()
        sql = clean_sql_string(raw_sql)

        print(f"\n📜 Generated SQL:\n{sql}\n")

        # Run SQL via pandasql
        try:
            result = pandasql.sqldf(sql, {"data": df})
            print(result.head(10))
        except Exception as e:
            print(f"⚠️  SQL execution error: {e}")

    except Exception as e:
        print(f"⚠️  Error generating SQL: {e}")

# Decision Simulation
@cmd("decision_simulation", "AI-driven decision simulator (usage: decision_simulation <scenario description>)")
def decision_simulation(df, *scenario_parts):
    if not check_data_loaded(df):
        return

    if not OPENAI_API_KEY:
        print("\u26a0\ufe0f No API key configured.")
        return

    scenario = " ".join(scenario_parts)
    if not scenario:
        print("Usage: decision_simulation <scenario description>")
        return

    from openai import OpenAI
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    schema_info = ", ".join(df.columns)
    sys_msg = (
        f"You are a data science assistant.\n"
        f"The user will describe a financial simulation scenario involving a DataFrame called 'data'.\n"
        f"Available columns in the data: {schema_info}\n\n"
        f"Your job is to:\n"
        f"1. Write a Python pandas code snippet to perform the simulation using this exact logic:\n"
        f"   - Apply the specified change to the attribute the user names (e.g., increase 'int_rate_n' by +5.0)\n"
        f"   - Adjust the 'Revenue' column based on the provided revenue impact formula, which the user will specify in the simulation description.\n"
        f"   - Compute the total of both original revenue and new revenue.\n"
        f"   - Compute the total impact (difference).\n"
        f"   - At the end of your code, assign a plain text result string to a variable called result_summary summarizing:\n"
        f"       - Original total revenue\n"
        f"       - New total revenue\n"
        f"       - The impact (delta)\n\n"
        f"2. Only return the Python code inside a markdown code block (```python ... ```).\n"
        f"Do not include explanations, placeholders, comments, or summaries outside the code.\n"
        f"Do not invent new formulas or assumptions — strictly follow the provided impact calculation formula.\n\n"
        f"If any required column is missing, raise a ValueError in the code.\n"
        f"End of instructions."
    )

    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": scenario}
            ],
            max_tokens=400,
            temperature=0
        )
        reply = response.choices[0].message.content.strip()

        import re
        code_match = re.search(r"```python(.*?)```", reply, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()

            local_scope = {"data": df.copy()}
            exec(code, {}, local_scope)

            result = local_scope.get("result_summary", "\u26a0\ufe0f No result_summary returned by the simulation.")

            print("📊  Simulation Result:\n")
            print(result)

        else:
            print("⚠️  No Python code block found in response.")

    except Exception as e:
        print(f"⚠️  Error generating simulation: {e}")


# Tree Explorer
@cmd("explore_tree", "Train & explore decision tree")
def explore_tree(df, target=None):
    if not target:
        print("Usage: explore_tree <target>")
        return
    if not check_data_loaded(df):
        return
    if not check_column_exists(df, target):
        return
    if pd.api.types.is_numeric_dtype(df[target]):
        print(f"⚠️  Column '{target}' is numeric. DecisionTreeClassifier requires a categorical (discrete) target.")
        print("Consider binning it into categories or using a regression tree if appropriate.")
        return

    num_cols = df.select_dtypes("number").drop(columns=[target], errors="ignore")
    X, y = num_cols, df[target]

    if y.nunique() < 2:
        print("⚠️  Need at least 2 target classes for classification.")
        return
    try:
        tree = DecisionTreeClassifier(max_depth=3)
        tree.fit(X, y)
        from sklearn.tree import export_text
        print(export_text(tree, feature_names=list(X.columns)))
    except Exception as e:
        print(f"⚠️  Error training decision tree: {e}")


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
        list_commands()
        return

    # manual override: allow user to type command directly
    tok = user_text.split()
    if not tok:
        print("⚠️ Not sure what you meant. Type 'help' to see available commands.")
        return

    if tok[0] in COMMANDS:
        _execute(df, tok)
        return

    # ---------- use GPT -----------------------------------------
    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY missing.  "
              "Type an exact command name or set the key."); return
    from openai import OpenAI
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    choices = "\n".join([f"- {k}: {d}" for k,(_,d) in COMMANDS.items()])
    sys_msg = (
        "You are an assistant that maps a user's request onto exactly ONE "
        "command name from the list below. Reply ONLY with that command "
        "name and, if required, its argument separated by space. Nothing else.\n\n"
        f"Commands:\n{choices}"
    )
    
    try:
        r = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role":"system","content":sys_msg},
                  {"role":"user",  "content":user_text}],
        max_tokens=20,
        temperature=0
        )
        cmd_line = r.choices[0].message.content.strip()
        _execute(df, cmd_line)
    except Exception as e:
        print(f"⚠️ Error processing command: {e}")
        print("Type 'help' to see available commands.")

def check_column_exists(df, col):
    if col not in df.columns:
        print(f"⚠️ Column '{col}' not found in data.")
        print(f"Available columns: {', '.join(df.columns)}")
        return False
    return True

def check_data_loaded(df):
    if df.empty:
        print("\u26a0\ufe0f  No data loaded.")
        return False
    return True

def _execute(df, name_and_args):
    if isinstance(name_and_args, str):
        parts = name_and_args.split()
    elif isinstance(name_and_args, list):
        parts = name_and_args
    else:
        print("Invalid command format.")
        return

    if not parts:
        print("No command found. Type 'help' to see available commands.")
        return

    name, *args = parts

    if name not in COMMANDS:
        print(f"⚠️ Unknown command: '{name}'. Type 'help' to list available commands.")
        return

    func = COMMANDS[name][0]
    print(f"\n▶ {name} {' '.join(args)}\n" + "-"*50)
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
