from sqlalchemy import create_engine, text
import os

POSTGRES_URL = os.environ.get("POSTGRES_URL") or     f"postgresql+psycopg2://{os.environ.get('POSTGRES_USER','argo')}:{os.environ.get('POSTGRES_PASSWORD','argo')}@{os.environ.get('POSTGRES_HOST','localhost')}:{os.environ.get('POSTGRES_PORT','5432')}/{os.environ.get('POSTGRES_DB','argo')}"

def main():
    engine = create_engine(POSTGRES_URL, future=True)
    with engine.begin() as conn:
        with open("db/schema.sql", "r", encoding="utf-8") as f:
            conn.execute(text(f.read()))
    print("✅ Tables ensured.")

if __name__ == "__main__":
    main()
