#!/usr/bin/env python
# lendingclub_eda.py
# ---------------------------------------------------------------
# One-file exploratory tool for Lending Club (or any tabular data)
# ---------------------------------------------------------------

import os, json, textwrap, inspect, functools
from collections import defaultdict
from pathlib import Path
import pandas as pd
import numpy as np
from dotenv import load_dotenv

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
        6. text_to_sql <describe the query in plain english>
        7. simulate_decision <describe the simulation scenario in plain english>
        8. explore_tree <target>
        9. impute_missing
        10.chart_grid <hist|scatter|bar|box > [target]
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
        return f # return functools.wraps(f)(f)
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

# Column Distribution 
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
@cmd("simulation_copilot", "AI-driven simulation copilot (usage: simulation_copilot <scenario description>)")
def simulation_copilot(df, *scenario_parts):
    if not check_data_loaded(df):
        return

    if not OPENAI_API_KEY:
        print("⚠️ No API key configured.")
        return

    scenario = " ".join(scenario_parts)
    if not scenario:
        print("Usage: simulation_copilot <scenario description>")
        return

    from openai import OpenAI
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    schema_info = ", ".join(df.columns)
    sys_msg = (
        f"You are a data science assistant.\n"
        f"The user will describe a simulation scenario in plain English.\n"
        f"Your job is to:\n"
        f"1. Write a Python pandas code snippet that performs the simulation on the provided DataFrame named 'data'.\n"
        f"2. Run the simulation and calculate key portfolio-level metrics.\n"
        f"3. Summarize the result in plain English.\n\n"
        f"Available columns: {schema_info}\n\n"
        f"Always return the Python code in a markdown code block (```python ... ```), then the summary.\n"
        f"Return nothing else."
    )

    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": scenario}
            ],
            max_tokens=600,
            temperature=0
        )
        reply = response.choices[0].message.content.strip()

        print("\n🤖 Simulation Copilot Response:\n")
        print(reply)

        # Extract code block
        import re
        code_match = re.search(r"```python(.*?)```", reply, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()

            print("\n📜 Executing generated code...\n")
            local_scope = {"data": df.copy()}
            exec(code, {}, local_scope)

        else:
            print("⚠️ No Python code found in response.")

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
        from sklearn.tree import DecisionTreeClassifier
        tree = DecisionTreeClassifier(max_depth=3)
        tree.fit(X, y)
        from sklearn.tree import export_text
        print(export_text(tree, feature_names=list(X.columns)))
    except Exception as e:
        print(f"⚠️  Error training decision tree: {e}")

# Impute missing values
@cmd("impute_missing", "Impute missing values: mean/median for numeric, mode for categorical — and show imputed value")
def impute_missing(df):
    if not check_data_loaded(df):
        return

    num_cols = df.select_dtypes(include="number").columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    impute_report = []

    for col in df.columns[df.isnull().any()]:
        if col in num_cols:
            # Check for outliers via IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower) | (df[col] > upper)]

            if outliers.empty:
                fill_value = round(df[col].mean(), 2)
                method = "mean"
            else:
                fill_value = round(df[col].median(), 2)
                method = "median"

            df[col] = df[col].fillna(fill_value)
            impute_report.append(f"{col} (numeric) → {method} ({fill_value})")

        elif col in cat_cols:
            fill_value = df[col].mode().iloc[0]
            df[col] = df[col].fillna(fill_value)
            impute_report.append(f"{col} (categorical) → mode ('{fill_value}')")

        else:
            impute_report.append(f"{col} → skipped (unsupported dtype)")

    print("\n📊  Missing value imputation completed.\n")
    for line in impute_report:
        print("•", line)

@cmd("chart_grid", "Automated chart grid (usage: chart_grid <hist|scatter|bar|box > [target])")
def chart_grid(df, chart_type="hist", target=None):
    if not check_data_loaded(df):
        return
    
    import seaborn as sns
    import matplotlib.pyplot as plt

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if chart_type == "hist":
        n_cols = 3
        n_rows = (len(num_cols) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = axes.flatten()

        for ix, col in enumerate(num_cols):
            df[col].hist(ax=axes[ix], bins=20)
            axes[ix].set_title(col)

        for ax in axes[len(num_cols):]:
            ax.set_visible(False)

        plt.tight_layout()
        plt.show()

    elif chart_type == "box":
        if target:
            if target not in df.columns:
             print(f"⚠️  Column '{target}' not found.")
             return
            for col in num_cols:
                plt.figure(figsize=(7, 4))
                sns.boxplot(x=target, y=col, data=df)
                plt.title(f"{col} by {target}")
                plt.show()
        else:
            for col in num_cols:
                plt.figure(figsize=(6, 4))
                sns.boxplot(y=col, data=df)
                plt.title(f"{col} distribution")
                plt.show()

    elif chart_type == "scatter":
        from itertools import combinations
        scatter_pairs = list(combinations(num_cols, 2))
        if len(scatter_pairs) > 10:
            scatter_pairs = scatter_pairs[:10]  # limit to 10 pairs to avoid clutter
            print("⚠️  Too many numeric columns, showing only first 10 scatterplots.\n")

        for x, y in scatter_pairs:
            plt.figure(figsize=(6, 4))
            sns.scatterplot(x=x, y=y, data=df)
            plt.title(f"{x} vs {y}")
            plt.show()

    elif chart_type == "bar":
        for col in cat_cols:
            plt.figure(figsize=(7, 4))
            df[col].value_counts(dropna=False).plot.bar()
            plt.title(f"Distribution of {col}")
            plt.xticks(rotation=45)
            plt.show()

    else:
        print("⚠️  Unknown chart type. Choose from: hist, box, scatter, bar")

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
