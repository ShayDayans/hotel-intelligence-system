"""
Databricks Configuration Module

Handles environment detection, secret management, and Azure Blob Storage configuration.
Works seamlessly in both local development and Databricks environments.
"""

import os


def is_databricks() -> bool:
    """
    Check if code is running in a Databricks environment.
    
    Returns:
        True if running on Databricks, False otherwise.
    """
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def get_secret(key: str) -> str:
    """
    Get secret/API key from environment.
    
    Works in both environments:
    - Databricks: Uses cluster environment variables
    - Local: Uses .env file via python-dotenv
    
    Args:
        key: Name of the secret/environment variable
        
    Returns:
        The secret value as a string
    """
    if is_databricks():
        # On Databricks, use environment variables set on the cluster
        value = os.environ.get(key, "")
        if not value:
            print(f"[WARNING] Secret '{key}' not found in cluster environment variables")
        return value
    else:
        # Local development - use .env file
        from dotenv import load_dotenv
        load_dotenv()
        return os.getenv(key, "")


def get_azure_storage_config() -> dict:
    """
    Return Azure Blob Storage configuration for Airbnb data.
    
    Returns:
        Dictionary with storage account details and paths
    """
    return {
        "storage_account": "lab94290",
        "airbnb_container": "airbnb",
        "airbnb_path": "airbnb_1_12_parquet",
        "sas_token": "sp=rle&st=2025-12-24T17:37:04Z&se=2026-02-28T01:52:04Z&spr=https&sv=2024-11-04&sr=c&sig=a0lx%2BS6PuS%2FvJ9Tbt4NKdCJHLE9d1Y1D6vpE1WKFQtk%3D"
    }


def configure_spark_for_azure(spark):
    """
    Configure Spark session for Azure Blob Storage access using SAS token.
    
    Args:
        spark: SparkSession instance
        
    Returns:
        Configured SparkSession
    """
    config = get_azure_storage_config()
    storage_account = config["storage_account"]
    sas_token = config["sas_token"].lstrip('?')
    
    spark.conf.set(
        f"fs.azure.account.auth.type.{storage_account}.dfs.core.windows.net", 
        "SAS"
    )
    spark.conf.set(
        f"fs.azure.sas.token.provider.type.{storage_account}.dfs.core.windows.net", 
        "org.apache.hadoop.fs.azurebfs.sas.FixedSASTokenProvider"
    )
    spark.conf.set(
        f"fs.azure.sas.fixed.token.{storage_account}.dfs.core.windows.net", 
        sas_token
    )
    
    print(f"[CONFIG] Spark configured for Azure Storage: {storage_account}")
    return spark


def get_airbnb_data_path() -> str:
    """
    Get the appropriate Airbnb data path based on environment.
    
    Returns:
        - Databricks: Azure Blob Storage path (abfss://...)
        - Local: Local file path
    """
    if is_databricks():
        config = get_azure_storage_config()
        storage_account = config["storage_account"]
        container = config["airbnb_container"]
        path = config["airbnb_path"]
        return f"abfss://{container}@{storage_account}.dfs.core.windows.net/{path}"
    else:
        return "data/sampled_airbnb_data.parquet"


def get_spark_session():
    """
    Get or create a properly configured Spark session.
    
    Returns:
        SparkSession configured for the current environment
    """
    from pyspark.sql import SparkSession
    
    if is_databricks():
        # On Databricks, get existing session and configure Azure
        spark = SparkSession.builder.getOrCreate()
        return configure_spark_for_azure(spark)
    else:
        # Local development
        import sys
        os.environ['PYSPARK_PYTHON'] = sys.executable
        os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
        
        return SparkSession.builder \
            .appName("HotelIntelligence") \
            .master("local[*]") \
            .config("spark.ui.showConsoleProgress", "false") \
            .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .config("spark.sql.parquet.enableVectorizedReader", "false") \
            .getOrCreate()


def print_environment_info():
    """Print current environment configuration for debugging."""
    print("=" * 50)
    print("ENVIRONMENT CONFIGURATION")
    print("=" * 50)
    print(f"Environment: {'Databricks' if is_databricks() else 'Local'}")
    
    if is_databricks():
        print(f"Runtime Version: {os.environ.get('DATABRICKS_RUNTIME_VERSION', 'Unknown')}")
    
    print(f"\nSecrets Status:")
    secrets = ["PINECONE_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY", "BRIGHTDATA_API_TOKEN"]
    for secret in secrets:
        value = get_secret(secret)
        status = "[OK]" if value else "[MISSING]"
        print(f"  {secret}: {status}")
    
    print(f"\nData Path:")
    print(f"  Airbnb: {get_airbnb_data_path()}")
    print("=" * 50)


# Quick test when run directly
if __name__ == "__main__":
    print_environment_info()
