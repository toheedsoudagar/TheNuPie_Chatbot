# db_setup.py
import os
import re
from pathlib import Path
from sqlalchemy import create_engine, text
import pandas as pd

# ---------- Configuration ----------
DOCS_DIR = "docs"
ANCHOR_DB_FILE = "agent_data.db"
ANCHOR_DB_URI = f"sqlite:///{ANCHOR_DB_FILE}"
# -----------------------------------

def _sanitize_mysql_for_sqlite(sql_script: str) -> str:
    """Cleans MySQL specific syntax to work with SQLite."""
    script = re.sub(r'/\*!.*?\*/;', '', sql_script, flags=re.DOTALL)
    script = re.sub(r'^--.*$', '', script, flags=re.MULTILINE)
    script = script.replace('`', '')
    script = re.sub(r'(tiny|small|medium|big)?int\s*\(\s*\d+\s*\)', 'INTEGER', script, flags=re.IGNORECASE)
    script = re.sub(r'\bdouble\b', 'REAL', script, flags=re.IGNORECASE)
    script = re.sub(r'\bfloat\b', 'REAL', script, flags=re.IGNORECASE)
    script = re.sub(r'\)\s*(ENGINE|AUTO_INCREMENT|DEFAULT CHARSET)=[^;]*;', ');', script, flags=re.IGNORECASE)
    script = re.sub(r'(LOCK|UNLOCK) TABLES.*?;', '', script, flags=re.IGNORECASE)
    return script

def _clean_col_name(col_name: str) -> str:
    """
    Normalizes a column name to snake_case.
    'Student Name' -> 'student_name'
    'Year (2020)' -> 'year_2020'
    """
    if not col_name: return "col"
    
    # 1. Convert to string and lowercase
    clean = str(col_name).lower().strip()
    
    # 2. Replace everything that isn't a-z or 0-9 with an underscore
    clean = re.sub(r'[^a-z0-9]', '_', clean)
    
    # 3. Collapse multiple underscores (e.g., "a___b" -> "a_b")
    clean = re.sub(r'_+', '_', clean)
    
    # 4. Strip leading/trailing underscores
    clean = clean.strip('_')
    
    # 5. Handle empty case (if column was just "!!!")
    if not clean: return "unnamed_col"
    
    return clean

def create_database_from_sql_files(db_uri: str = ANCHOR_DB_URI):
    docs_path = Path(DOCS_DIR)
    
    # --- 1. Create the Anchor Database ---
    if os.path.exists(ANCHOR_DB_FILE):
        try: os.remove(ANCHOR_DB_FILE)
        except Exception: pass
            
    anchor_engine = create_engine(ANCHOR_DB_URI)
    with anchor_engine.connect() as conn:
        conn.execute(text("CREATE TABLE IF NOT EXISTS _agent_metadata (id INTEGER PRIMARY KEY, info TEXT);"))
        conn.commit()
    print(f"[DB Setup] Created anchor database: {ANCHOR_DB_FILE}")

    # --- 2. Process Files ---
    all_files = sorted([
        f for f in docs_path.iterdir() 
        if f.suffix.lower() in ['.sql', '.csv', '.xlsx', '.xls']
    ])

    if not all_files:
        print("[DB Setup] No data files found in docs/.")
        return ANCHOR_DB_URI

    for file_path in all_files:
        specific_db_name = file_path.stem + ".db"
        specific_db_path = docs_path / specific_db_name
        
        # Ensure fresh start for every DB
        if specific_db_path.exists():
            try: os.remove(specific_db_path)
            except Exception: pass

        specific_uri = f"sqlite:///{specific_db_path}"
        engine = create_engine(specific_uri)
        print(f"[DB Setup] Processing: {file_path.name} -> {specific_db_name}")

        try:
            # A. Handle SQL Files
            if file_path.suffix.lower() == '.sql':
                with engine.connect() as conn:
                    raw = file_path.read_text(encoding='utf-8-sig')
                    clean = _sanitize_mysql_for_sqlite(raw)
                    statements = re.split(r';\s*$', clean, flags=re.MULTILINE)
                    count = 0
                    for stmt in statements:
                        if stmt.strip():
                            try:
                                conn.execute(text(stmt.strip()))
                                count += 1
                            except Exception: pass
                    conn.commit()
                print(f"    - Executed {count} SQL statements.")

            # B. Handle CSV/Excel (With Column Normalization)
            elif file_path.suffix.lower() in ['.csv', '.xlsx', '.xls']:
                # Read Data
                if file_path.suffix.lower() == '.csv':
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)
                
                # --- NORMALIZE COLUMNS ---
                old_cols = list(df.columns)
                df.columns = [_clean_col_name(c) for c in df.columns]
                new_cols = list(df.columns)
                
                # Sanitize Table Name
                table_name = _clean_col_name(file_path.stem)
                
                # Write to DB
                df.to_sql(table_name, engine, index=False, if_exists='replace')
                print(f"    - Table '{table_name}' created ({len(df)} rows).")
                print(f"    - Columns normalized (e.g., '{old_cols[0]}' -> '{new_cols[0]}')")

        except Exception as e:
            print(f"    [!] Error processing {file_path.name}: {e}")

    print("[DB Setup] Database generation complete.")
    return ANCHOR_DB_URI

if __name__ == "__main__":
    create_database_from_sql_files(ANCHOR_DB_URI)