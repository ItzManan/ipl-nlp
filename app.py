from langchain_community.utilities import SQLDatabase
import os
from dotenv import load_dotenv
import gradio as gr

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

db = SQLDatabase.from_uri(DATABASE_URL)

from typing_extensions import TypedDict

class State(TypedDict):
    question: str
    expanded_question: str
    query: str
    result: str
    answer: str


from langchain.chat_models import init_chat_model

llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

from langchain_core.prompts import ChatPromptTemplate

system_message = """
You are an expert in IPL cricket statistics and PostgreSQL SQL generation. Your task is to take natural language questions about IPL performance and generate highly accurate SQL queries using only the provided database schema and structure.

You must ensure:
- Queries are compatible with the {dialect} SQL dialect (PostgreSQL).
- Queries should never return more than {top_k} rows unless explicitly requested.
- Only include the **most relevant columns**, based on the question.
- Never use `SELECT *`. Always specify only necessary columns.
- Be cautious of table and column names—only use what's defined in the schema.
- Pay special attention to which player belongs to which team in a match.
- Use single quotes when filtering on strings (e.g., player or team names).

---

🏏 IPL-Specific Logic and Context:

Your schema includes detailed IPL data including:
- `players`: all IPL players
- `teams`: all IPL franchises
- `matches`: one row per match with metadata like winner, season, batting order
- `player_matches`: one row per player per match with stats like runs, wickets, sixes, etc.
- `venues`: all IPL venues
- `deliveries`: ball-by-ball data for each match
- `player_teams`: player-team relationships

You should follow these IPL-specific guidelines:

1. **Chasing or Batting Second Players:**
   Use `player_matches.team_id = matches.batting_second_team_id`

2. **Batting First Players:**
   Use `player_matches.team_id = matches.batting_first_team_id`

3. **Winning Team Players:**
   Use `player_matches.team_id = matches.winner_id`

4. **Losing Team Players:**
   Use `player_matches.team_id != matches.winner_id`

5. **Filter by Player or Team Name:**
   Always use:  
   `WHERE players.name = 'Virat Kohli'`  
   or  
   `WHERE teams.name = 'Mumbai Indians'`

6. **Season-Specific Questions:**
   Use `matches.season = 2024` or the required year.

---

📛 Naming Standards — STRICTLY FOLLOW:

✅ Always use standardized player and team names as stored in the database.

✅ For teams, always use full official names. For example:
- `'Royal Challengers Bengaluru'` (NOT `'Bangalore'`)
- `'Punjab Kings'` (NOT `'Kings XI Punjab'`)
- `'Delhi Capitals'` (NOT `'Delhi Daredevils'`)

✅ For players, ensure full spelling and case match, e.g. `'MS Dhoni'`, `'Virat Kohli'`, `'Rinku Singh'`

Do NOT assume alternate spellings will match.

In queries like batting first or chasing, do not assume the player is in the winning team. Always check the `matches` table for the correct team.

If season is not specified, assume whole IPL history for the player.

---

✅ Example 1 (Player-based stat in a season):
"How many sixes did Rinku Singh hit in IPL 2023?"  
```
SELECT SUM(T1.sixes) 
FROM player_matches AS T1 
JOIN players AS T2 ON T1.player_id = T2.id 
JOIN matches AS T3 ON T1.match_id = T3.id 
WHERE T2.name = 'Rinku Singh' AND T3.season = 2023;
```

✅ Example 2 (Chasing team performance):
"Top 3 six-hitters in successful chases over 180 runs"

```
SELECT T2.name, SUM(T1.sixes) AS total_sixes 
FROM player_matches AS T1 
JOIN players AS T2 ON T1.player_id = T2.id 
JOIN matches AS T3 ON T1.match_id = T3.id 
WHERE T3.batting_second_runs >= 180 
  AND T3.winner_id = T3.batting_second_team_id 
  AND T1.team_id = T3.batting_second_team_id 
GROUP BY T2.id, T2.name 
ORDER BY total_sixes DESC 
LIMIT 3;
```

🧠 Additional Tips:

Always join only necessary tables: player_matches, players, matches, teams

Do not hallucinate columns—only use ones present in the schema

Use joins correctly to relate player stats to match context

Only use the following tables and columns:
{table_info}
"""

user_prompt = "Question: {input}"

query_prompt_template = ChatPromptTemplate(
    [("system", system_message), ("user", user_prompt)]
)

from pydantic import BaseModel, Field

class QueryOutput(BaseModel):
    query: str = Field(..., description="Syntactically valid SQL query.")


def natural_language_expand(state: State):
    prompt = (
f"""You are an assistant that rewrites vague or short cricket database queries into detailed, unambiguous natural language.
Expand and clarify the following user query into a complete and clear bullet points that can be understood by a SQL generator.
A bowler or a batter be referred to as a player.
User Query: {state["question"]}"""
    )
    response = llm.invoke(prompt)
    return {"expanded_question": response.content}

def write_query(state: State):
    """Generate SQL query to fetch information."""
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": state["expanded_question"],
        }
    )
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return {"query": result.query}


from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

def execute_query(state: State):
    """Execute SQL query."""
    execute_query_tool = QuerySQLDataBaseTool(db=db)
    return {"result": execute_query_tool.invoke(state["query"])}

def generate_answer(state: State):
    """Answer question using retrieved information as context."""
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}'
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}

from langgraph.graph import START, StateGraph

graph_builder = StateGraph(State).add_sequence(
    [natural_language_expand, write_query, execute_query, generate_answer]
)
graph_builder.add_edge(START, "natural_language_expand")
graph = graph_builder.compile()


def process_question(question: str):
    try:
        state: State = {"question": question, "expanded_question": "", "query": "", "result": "", "answer": ""}
        result = graph.invoke(state)

        return (
            result.get("answer", ""),
            "🧠 Expanded Question\n\n" + result.get("expanded_question", ""),
            result.get("query", ""),
            result.get("result", "No result found"),
        )

    except Exception as e:
        return f"❌ Error: {str(e)}", "", "", ""

# Gradio UI
with gr.Blocks(title="🏏 IPL Natural Language Stats Explorer") as demo:
    gr.Markdown(
        """
        # 🏏 IPL Natural Language Stats Explorer
        Ask IPL-related questions in plain English. The system will convert it to SQL, run it, and give you an answer.
        """
    )
    with gr.Row():
        with gr.Column():
            question_input = gr.Textbox(
                label="Ask an IPL Question", 
                placeholder="e.g. Most sixes in IPL 2023 by a player"
            )
            submit_button = gr.Button("Submit", variant="primary")
            database_schema_link = gr.Markdown("Check out the Database Schema in detail [here](https://dbdocs.io/sanchitjain1107/ipl-database?view=relationships)")
            database_schema = gr.Image("ipl-database-erd.png", label="Database Schema")
        with gr.Column():
            final_answer = gr.Textbox(label="📋 Final Answer")
            expanded_question = gr.Markdown(container=True, value="🧠 Expanded Question")
            sql_query = gr.Textbox(label="📄 SQL Query")
            query_result = gr.Textbox(label="📊 Query Result")

    submit_button.click(
        fn=process_question,
        inputs=[question_input],
        outputs=[final_answer, expanded_question, sql_query, query_result]
    )

demo.launch()
