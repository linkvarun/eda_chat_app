The tool is an interactive, command-driven Exploratory Data Analysis (EDA) assistant that combines human readable commands in natural language, LLM driven copilot features (like text-to-SQL and simulation scenarios) 
and automated charting grid — all in a single light weight application for quick data analysis. It has a balanced combination of deteministic and stochastic functions to ensure it performs the intended operations
only and doesn't stray or hallucinate.

Technical Design:
  It has a CLI driven interface which offers a simple prompt based intercation. 
  The commands are stored in a COMMAND dictionary for easy disptach.  
  Gemini API key (via OPENAI) is used for understanding and generating natural language.
  Keeps a single in-memory dataframe df as the shared object for all operations.
  The metadata/schema is passed to LLM and the sql or python code generated (e.g. by text_to_sql or simulate_decision) then runs inside a local environment .
  Schema is inferred by reading a few samples of data.
  matplotlib and seaborn are used for creating visulaizations like charts grids, heatmap.
  pandassql is used with LLM generated SQL, to execute on the data.
  Graceful fallback for nearly all setbacks.

User functionality:


Future Extensions:
1) Memory
2) Interface
3) More functions
4) Fine tuning
