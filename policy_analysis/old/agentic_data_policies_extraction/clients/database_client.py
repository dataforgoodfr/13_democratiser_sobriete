import logging
import os
from typing import Any, Dict, List, Tuple, Union

from sqlalchemy import text
from sqlmodel import Session, create_engine

# Configure logging
logger = logging.getLogger(__name__)

# Database configuration - same as persist_taxonomy.py
DATABASE_URL = os.getenv("DATABASE_URL")

# Create database engine - same pattern as persist_taxonomy.py
db_engine = create_engine(DATABASE_URL, pool_pre_ping=True)


class DatabaseClient:
    """SQL client for connecting to the main PostgreSQL database using SQLModel/SQLAlchemy"""

    def __init__(self, engine=None):
        """
        Initialize database client

        Args:
            engine: SQLAlchemy engine. If None, uses the default engine.
        """
        self.engine = engine or db_engine

    def execute_query(
        self, query: str, params: Union[Dict, Tuple, None] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a SQL query and return results as a list of dictionaries

        Args:
            query: SQL query string
            params: Query parameters (dict for named params, tuple for positional)

        Returns:
            List of dictionaries containing query results
        """
        try:
            with Session(self.engine) as session:
                # Execute the query using SQLAlchemy text()
                result = session.execute(text(query), params or {})

                # Convert results to list of dictionaries
                if result.returns_rows:
                    columns = result.keys()
                    return [dict(zip(columns, row)) for row in result.fetchall()]
                else:
                    return []
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise

    def list_tables(self) -> List[str]:
        """
        List all tables in the database

        Returns:
            List of table names
        """
        query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name;
        """

        result = self.execute_query(query)
        return [row["table_name"] for row in result]

    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """
        Get information about a table structure

        Args:
            table_name: Name of the table

        Returns:
            Dictionary containing table information
        """
        query = """
        SELECT 
            column_name,
            data_type,
            is_nullable,
            column_default
        FROM information_schema.columns 
        WHERE table_name = :table_name
        ORDER BY ordinal_position;
        """

        columns = self.execute_query(query, {"table_name": table_name})

        # Get row count
        count_query = f"SELECT COUNT(*) as count FROM {table_name}"
        count_result = self.execute_query(count_query)
        row_count = count_result[0]["count"] if count_result else 0

        return {"table_name": table_name, "columns": columns, "row_count": row_count}

    def query_table(
        self, table_name: str, limit: int = 10, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Query any table in the database

        Args:
            table_name: Name of the table to query
            limit: Number of rows to return
            offset: Number of rows to skip

        Returns:
            List of dictionaries containing table data
        """
        query = f"""
        SELECT * FROM {table_name}
        ORDER BY 1
        LIMIT :limit OFFSET :offset
        """

        return self.execute_query(query, {"limit": limit, "offset": offset})

    def search_table(
        self, table_name: str, search_term: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search any table by text content in string columns

        Args:
            table_name: Name of the table to search
            search_term: Text to search for
            limit: Number of results to return

        Returns:
            List of dictionaries containing matching data
        """
        # First get column info to find text columns
        table_info = self.get_table_info(table_name)
        text_columns = [
            col["column_name"]
            for col in table_info["columns"]
            if "char" in col["data_type"].lower() or "text" in col["data_type"].lower()
        ]

        if not text_columns:
            logger.warning(f"No text columns found in table {table_name}")
            return []

        # Build dynamic search query
        search_conditions = " OR ".join(
            [f"{col} ILIKE :search_pattern" for col in text_columns]
        )
        query = f"""
        SELECT * FROM {table_name}
        WHERE {search_conditions}
        ORDER BY 1
        LIMIT :limit
        """

        search_pattern = f"%{search_term}%"
        return self.execute_query(query, {"search_pattern": search_pattern, "limit": limit})

    def query_policies_abstracts_all(
        self, limit: int = 10, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Query the policies_abstracts_all table

        Args:
            limit: Number of rows to return
            offset: Number of rows to skip

        Returns:
            List of dictionaries containing policy data
        """
        query = """
        SELECT * FROM public.policies_abstracts_all 
        ORDER BY openalex_id
        LIMIT :limit OFFSET :offset
        """

        return self.execute_query(query, {"limit": limit, "offset": offset})

    def save_extraction_results(
        self, openalex_id: str, extraction_data: Dict[str, Any], conclusion: str
    ) -> bool:
        """
        Save extraction results to the database

        Args:
            openalex_id: OpenAlex ID of the policy
            extraction_data: Extracted data from AI system
        """
        # Insert the extraction data using execute_query
        insert_query = """
        INSERT INTO agentic_policy_extractions (openalex_id, extraction_data, conclusion)
        VALUES (:openalex_id, :extraction_data, :conclusion)
        ON CONFLICT (openalex_id) 
        DO UPDATE SET
            extraction_data = EXCLUDED.extraction_data,
            conclusion = EXCLUDED.conclusion,
            updated_at = CURRENT_TIMESTAMP
        """

        # Execute the insert query
        result = self.execute_query(
            insert_query,
            {
                "openalex_id": openalex_id,
                "conclusion": conclusion,
                "extraction_data": extraction_data,
            },
        )

    def create_policy_extractions_table(self):
        """
        Create the policy_extractions table
        """
        query = """
        CREATE TABLE IF NOT EXISTS agentic_policy_extractions (
            id SERIAL PRIMARY KEY,
            openalex_id VARCHAR(255) NOT NULL,
            extraction_data JSONB,
            conclusion TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        self.execute_query(query)
