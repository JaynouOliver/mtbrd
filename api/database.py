"""
Database connection pool for vector search API.
Uses asyncpg for high-performance async PostgreSQL connections.
"""
import asyncpg
import os
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)


class DatabasePool:
    """Manages connection pools for primary and replica databases."""
    
    def __init__(self):
        self.primary_pool = None
        self.replica_pool = None
        
    async def connect(self):
        """Initialize connection pools."""
        # Build connection URLs from individual env vars if DATABASE_URL not provided
        primary_url = os.getenv("DATABASE_URL_PRIMARY")
        replica_url = os.getenv("DATABASE_URL_REPLICA")
        
        # Fallback to individual env vars (existing format)
        if not primary_url:
            user = os.getenv("user")
            password = os.getenv("password")
            host = os.getenv("host")
            port = os.getenv("port", "6543")
            dbname = os.getenv("dbname", "postgres")
            primary_url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}?sslmode=require"
        
        if not replica_url:
            # Use same as primary if replica not specified
            replica_url = primary_url
        
        logger.info(f"Connecting to primary: {primary_url.split('@')[1] if '@' in primary_url else 'configured'}")
        logger.info(f"Connecting to replica: {replica_url.split('@')[1] if '@' in replica_url else 'configured'}")
        
        self.primary_pool = await asyncpg.create_pool(
            primary_url,
            min_size=2,
            max_size=5,
            command_timeout=30,
            timeout=3,
            ssl="require",
        )
        
        self.replica_pool = await asyncpg.create_pool(
            replica_url,
            min_size=5,
            max_size=25,
            command_timeout=30,
            timeout=3,
            ssl="require",
        )
        
        logger.info("Connection pools initialized")
    
    async def disconnect(self):
        """Close all connection pools."""
        if self.primary_pool:
            await self.primary_pool.close()
        if self.replica_pool:
            await self.replica_pool.close()
        logger.info("Pools closed")


db_pool = DatabasePool()

