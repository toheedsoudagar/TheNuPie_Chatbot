# sql_agent.py
import re
from sqlalchemy import create_engine, text
from langchain_ollama import ChatOllama
from pathlib import Path

class SQLAgent:
    def __init__(self, db_uri: str, llm_model: str, llm_temperature: float, base_url: str = None):
        self.db_uri = db_uri
        self.attached_dbs = [] 
        
        try:
            self.engine = create_engine(db_uri)
            with self.engine.connect() as conn: conn.execute(text("SELECT 1"))
            print("[SQL] Anchor database connected.")
        except Exception:
            self.engine = create_engine("sqlite:///:memory:")

        self._attach_external_databases()

        print(f"[SQL] Initializing LLM: {llm_model} (URL: {base_url})")
        self.llm = ChatOllama(
            model=llm_model, 
            temperature=llm_temperature,
            base_url=base_url
        )
        
        print("[SQL] Building Value-Aware Schema Map...")
        self.schema_map = self._build_value_aware_schema_map()

    def _attach_external_databases(self):
        docs_dir = Path("docs")
        if not docs_dir.exists(): return
        extra_dbs = [f for f in docs_dir.glob("*.db") if f.name != "agent_data.db"]
        
        print(f"[SQL] Found {len(extra_dbs)} external databases. Attaching...")
        with self.engine.connect() as conn:
            for db_file in extra_dbs:
                alias = db_file.stem.replace(" ", "_").replace("-", "_").replace(".", "_").lower()
                db_path = str(db_file.absolute()).replace("\\", "/")
                try:
                    conn.execute(text(f"ATTACH DATABASE '{db_path}' AS {alias}"))
                    self.attached_dbs.append(alias)
                except Exception: pass

    def _build_value_aware_schema_map(self):
        schema_info = []
        if not self.attached_dbs: return "(No tables found)"

        with self.engine.connect() as conn:
            for alias in self.attached_dbs:
                try:
                    tables = conn.execute(text(f"SELECT name FROM {alias}.sqlite_master WHERE type='table'")).fetchall()
                    for t_row in tables:
                        table = t_row[0]
                        if table.startswith("sqlite_"): continue
                        
                        cols = conn.execute(text(f"PRAGMA {alias}.table_info('{table}')")).fetchall()
                        col_details = []
                        for c in cols:
                            col_name = c[1]
                            example_vals = ""
                            try:
                                q = text(f"SELECT {col_name}, COUNT(*) as c FROM {alias}.{table} GROUP BY {col_name} ORDER BY c DESC LIMIT 3")
                                res = conn.execute(q).fetchall()
                                vals = [str(r[0]) for r in res if r[0] is not None]
                                if vals:
                                    clean_vals = [v[:15] + "..." if len(v) > 15 else v for v in vals]
                                    example_vals = f" (e.g. {', '.join(clean_vals)})"
                            except: pass
                            col_details.append(f"- {col_name}{example_vals}")

                        info = (f"TABLE: `{alias}.{table}`\nCOLUMNS:\n" + "\n".join(col_details))
                        schema_info.append(info)
                except Exception: pass
        return "\n---------------------\n".join(schema_info)

    def _clean_sql(self, sql_text: str) -> str:
        clean = re.sub(r"```sql|```", "", sql_text, flags=re.IGNORECASE).strip()
        match = re.search(r"\b(SELECT|WITH)\b", clean, re.IGNORECASE)
        if match: clean = clean[match.start():]
        clean = clean.replace("`", '"')
        clean = clean.rstrip(";") # Prevents "Multiple statements" error
        return clean

    def _generate_sql(self, query: str) -> str:
        prompt = (
            "You are an expert SQL Data Analyst. Write a SQLite query.\n"
            f"Schema:\n{self.schema_map}\n\n"
            "**CRITICAL RULES:**\n"
            "1. **Check Values:** Use provided examples. If 'BDS' is in `department`, use `department='BDS'`.\n"
            "2. **No Backticks:** Use double quotes for identifiers: `\"Student Name\"`.\n"
            "3. **Aggregation:** For 'top', 'popular', 'distribution', or 'breakdown', use `GROUP BY column ORDER BY COUNT(*) DESC`.\n"
            "4. **No Hallucinated Filters:** Do NOT add `WHERE` filters unless asked.\n"
            "5. **Output:** Return ONLY the SQL query.\n"
            f"Question: {query}"
        )
        response = self.llm.invoke(prompt)
        return self._clean_sql(response.content.strip())

    def _summarize_results(self, query: str, results: list) -> str:
        if not results: return "No results found."
        data_preview = str(results[:8])
        prompt = (
            "You are a Data Reporter. Report the data retrieved.\n"
            f"Question: {query}\nData: {data_preview}\n\n"
            "Instructions:\n1. Report ONLY what is in the Data.\n2. Do NOT invent extra items.\n3. Be concise."
        )
        return self.llm.invoke(prompt).content.strip()

    def ask(self, query: str):
        try:
            sql_query = self._generate_sql(query)
            print(f"[SQL Agent] Query: {sql_query}")
            
            used_source = "Unknown"
            for alias in self.attached_dbs:
                if alias in sql_query: used_source = f"{alias}.db"; break

            with self.engine.connect() as conn:
                if "limit" not in sql_query.lower(): sql_query += " LIMIT 20"
                
                result = conn.execute(text(sql_query))
                keys = result.keys()
                rows = result.fetchall()
                
            if not rows: return "No data found.", [], used_source
            
            structured_data = [dict(zip(keys, row)) for row in rows]
            print(f"[SQL Agent] Summarizing {len(rows)} rows...")
            summary = self._summarize_results(query, structured_data)

            return summary, structured_data, used_source

        except Exception as e:
            print(f"[SQL Error] {e}")
            return "Error executing query.", [], "Error"