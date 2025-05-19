The tool is an interactive, command-driven Exploratory Data Analysis (EDA) assistant that combines human readable commands in natural language, LLM driven copilot features (like text-to-SQL and simulation scenarios) 
and automated charting grid — all in a single light weight application for quick data analysis. It has a balanced combination of deteministic and stochastic functions to ensure it performs the intended operations
only and doesn't stray or hallucinate.

Technical Design:
  1) It has a CLI driven interface which offers a simple prompt based intercation. 
  2) The commands are stored in a COMMAND dictionary for easy disptach.  
  3) Gemini API key (via OPENAI) is used for understanding and generating natural language.
  4) Keeps a single in-memory dataframe df as the shared object for all operations.
  5) The metadata/schema is passed to LLM and the sql or python code generated (e.g. by text_to_sql or simulate_decision) then runs inside a local environment .
  6) Schema is inferred by reading a few samples of data.
  7) matplotlib and seaborn are used for creating visulaizations like charts grids, heatmap.
  8) For missing value imputaion, the tool checks for outliers in numerical values. If absent, it suggests mean and if present, it suggest median for imputaion. For categorical features, it suggets mode.
  9) For tree explore, it is currently limited to Classification Decision tree which means the target variable has to be categorical.
  10) pandassql is used with LLM generated SQL, to execute on the data.
  11) Graceful fallback for nearly all setbacks.

User functionality(first 5 functions were already present, I have added from function 6 to 10):
        1. missing_values-> Type missing_values and the tool finds the features with missing values and their respective %
        2. describe_numeric -> Type describe_numeric to get a description of the numeric features
        3. value_counts <column> -> Type value_counts follwed by the column to get the counts of the values of that categorical feature 
        4. correlation_heatmap -> Type correlation_heatmap to get the correlation heatmap of all numerical features
        5. class_balance (if a target column exists) -> Type class_balance followed by a column to get the bar chart visulaization of the value counts of that categorical feature 
        6. text_to_sql <describe the query in plain english> -> Type text_to_sql followed by the query that you want the tool to generate and execute, in plain english or natural language
        7. simulate_decision <describe the simulation scenario in plain english> -> Type simulate_decision followed by the decision simulation scenario like what's the impact of increase in int_rate_n on Revenue
        8. explore_tree <target> -> Type explore_tree followed by a categorical target variable and the tool will render the tree branches and bifurcation rules (upto max depth=3) of the Decision tree
        9. impute_missing -> Type impute_missing and the tool will suggest the appropriate values to be imputed to the features with missing values
        10.chart_grid <hist|scatter|bar|box > [target]  -> Type chart_grid followed by type of chart required (hist|scatter|box) for all numeric and bar for all categorical features. If no bar type is mentioned,
        by default histogram is created.


Future Extensions:
1) More features like Outlier detection and treatment, Feature Encoding, Feature Scaling, Auto ML, to be added.
2) Memory storage to retain the complete chain of conversation.
3) Add a graphical user interface with Streamlit or Django framework.
4) Expand the decision tree to include regression use cases, along with Decision tree classifier.
5) Export option to save the data and visualizations.
6) For missing value imputation, outlier treatment, encoding and scaling, provide user a choice to execute from the provided options (e.g. option to use label encoding or one-hot encoding; retain outliers, remove or replace them with a suggested or user provided reasonable value.
7) Fine tuning system prompts to generate the intented response (currently simulate_decision is not doing a great job). 
8) More visualizations to be added for better feature analysis.
9) Perform statistical tests like Chi-sq, ANNOVA F stat, VIF, find weights of evidence and Information values.
   
