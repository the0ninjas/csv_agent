from dotenv import load_dotenv
import os
import mysql.connector
from mysql.connector import Error
import pandas as pd
from datetime import datetime

load_dotenv()

class MySQLConnection:
    def __init__(self):
        self.config = {
            'host': os.getenv('MYSQL_HOST', 'localhost'),
            'port': int(os.getenv('MYSQL_PORT', 3306)),
            'database': os.getenv('MYSQL_DATABASE'),
            'user': os.getenv('MYSQL_USER'),
            'password': os.getenv('MYSQL_PASSWORD')
        }
        
    def get_connection(self):
        try:
            connection = mysql.connector.connect(**self.config)
            return connection
        except Error as e:
            print(f"Error connecting to MySQL: {e}")
            return None
    
    def execute_query(self, query, params=None):
        connection = self.get_connection()
        if connection:
            try:
                cursor = connection.cursor()
                cursor.execute(query, params)
                result = cursor.fetchall()
                cursor.close()
                connection.close()
                return result
            except Error as e:
                print(f"Error executing query: {e}")
                return None
        return None
    
    # def create_tables_for_langgraph(self):
    #     """Create tables that LangGraph might need for checkpointing"""
    #     connection = self.get_connection()
    #     if connection:
    #         try:
    #             cursor = connection.cursor()
                
    #             # Create a simple checkpoints table
    #             create_table_query = """
    #             CREATE TABLE IF NOT EXISTS langgraph_checkpoints (
    #                 id VARCHAR(255) PRIMARY KEY,
    #                 thread_id VARCHAR(255),
    #                 checkpoint_data JSON,
    #                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    #                 INDEX idx_thread_id (thread_id)
    #             );
    #             """
                
    #             cursor.execute(create_table_query)
    #             connection.commit()
    #             print("LangGraph tables created successfully!")
                
    #             cursor.close()
    #             connection.close()
    #         except Error as e:
    #             print(f"Error creating tables: {e}")

    def create_staging_table(self):
        """Create the article data table with specified schema"""
        connection = self.get_connection()
        if not connection:
            print("❌ Failed to connect to database for staging table creation")
            return False
            
        try:
            cursor = connection.cursor()
            
            # Check if table already exists
            cursor.execute("""
                SELECT COUNT(*) 
                FROM information_schema.tables 
                WHERE table_schema = %s 
                AND table_name = 'articles'
            """, (self.config['database'],))
            
            table_exists = cursor.fetchone()[0] > 0
            
            if table_exists:
                print("ℹ️ Staging table 'articles' already exists")
                cursor.close()
                connection.close()
                return True
            
            # Create article data table
            create_staging_table_query = """
            CREATE TABLE IF NOT EXISTS articles (
                ArticleID VARCHAR(255) PRIMARY KEY,
                ArtDate DATE,
                Month VARCHAR(20),
                Year INT,
                CompetName VARCHAR(255),
                KPMGTotalImpact FLOAT,
                DeloitteTotalImpact FLOAT,
                EYTotalImpact FLOAT,
                PwCTotalImpact FLOAT,
                Issue TEXT,
                Industry VARCHAR(255),
                Comments TEXT,
                SpokespersonName VARCHAR(255), 
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_date (ArtDate),
                INDEX idx_year (Year),
                INDEX idx_industry (Industry),
                INDEX idx_compet_name (CompetName)
            );
            """
            
            cursor.execute(create_staging_table_query)
            connection.commit()
            print("✅ Staging table created successfully!")
            
            cursor.close()
            connection.close()
            return True
            
        except Error as e:
            print(f"❌ Error creating staging table: {e}")
            if connection.is_connected():
                connection.rollback()
                connection.close()
            return False

    def create_article_table(self):
        """Create the article data table with specified schema"""
        connection = self.get_connection()
        if not connection:
            print("❌ Failed to connect to database for article table creation")
            return False
            
        try:
            cursor = connection.cursor()
            
            # Check if table already exists
            cursor.execute("""
                SELECT COUNT(*) 
                FROM information_schema.tables 
                WHERE table_schema = %s 
                AND table_name = 'articles'
            """, (self.config['database'],))
            
            table_exists = cursor.fetchone()[0] > 0
            
            if table_exists:
                print("ℹ️ Article table 'articles' already exists")
                cursor.close()
                connection.close()
                return True
            
            # Create article data table
            # Comments VECTOR dimension is 384 (all_minilm_l12_v2) as available in MySQL
            create_article_table_query = """
            CREATE TABLE IF NOT EXISTS articles (
                ArticleID VARCHAR(255) PRIMARY KEY,
                ArtDate DATE,
                Month VARCHAR(20),
                Year INT,
                CompetName VARCHAR(255),
                KPMGTotalImpact FLOAT,
                DeloitteTotalImpact FLOAT,
                EYTotalImpact FLOAT,
                PwCTotalImpact FLOAT,
                Issue TEXT,
                Industry VARCHAR(255),
                Comments TEXT,
                Comments_embedding VECTOR(384),
                SpokespersonName VARCHAR(255), 
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_date (ArtDate),
                INDEX idx_year (Year),
                INDEX idx_industry (Industry),
                INDEX idx_compet_name (CompetName)
            );
            """
            
            cursor.execute(create_article_table_query)
            connection.commit()
            print("✅ Article table created successfully!")
            
            cursor.close()
            connection.close()
            return True
            
        except Error as e:
            print(f"❌ Error creating article table: {e}")
            if connection.is_connected():
                connection.rollback()
                connection.close()
            return False
                
    def import_csv_to_staging(self, csv_file_path):
        """Import CSV data into the staging articles table (without embeddings)"""
        try:
            # Read CSV file
            df = pd.read_csv(csv_file_path)
            
            connection = self.get_connection()
            if connection:
                cursor = connection.cursor()
                
                # Prepare insert query for staging table
                insert_query = """
                INSERT INTO articles (
                    ArticleID, ArtDate, Month, Year, CompetName,
                    KPMGTotalImpact, DeloitteTotalImpact, EYTotalImpact, PwCTotalImpact,
                    Issue, Industry, Comments, SpokespersonName
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    ArtDate = VALUES(ArtDate),
                    Month = VALUES(Month),
                    Year = VALUES(Year),
                    CompetName = VALUES(CompetName),
                    KPMGTotalImpact = VALUES(KPMGTotalImpact),
                    DeloitteTotalImpact = VALUES(DeloitteTotalImpact),
                    EYTotalImpact = VALUES(EYTotalImpact),
                    PwCTotalImpact = VALUES(PwCTotalImpact),
                    Issue = VALUES(Issue),
                    Industry = VALUES(Industry),
                    Comments = VALUES(Comments),
                    SpokespersonName = VALUES(SpokespersonName)
                """
                
                # Insert data row by row
                successful_imports = 0
                for index, row in df.iterrows():
                    try:
                        # Convert date if needed
                        art_date = None
                        if pd.notna(row.get('ArtDate')):
                            try:
                                art_date = pd.to_datetime(row['ArtDate']).date()
                            except:
                                art_date = None
                        
                        data = (
                            row.get('ArticleID'),
                            art_date,
                            row.get('Month'),
                            int(row.get('Year')) if pd.notna(row.get('Year')) else None,
                            row.get('CompetName'),
                            float(row.get('KPMGTotalImpact')) if pd.notna(row.get('KPMGTotalImpact')) else None,
                            float(row.get('DeloitteTotalImpact')) if pd.notna(row.get('DeloitteTotalImpact')) else None,
                            float(row.get('EYTotalImpact')) if pd.notna(row.get('EYTotalImpact')) else None,
                            float(row.get('PwCTotalImpact')) if pd.notna(row.get('PwCTotalImpact')) else None,
                            row.get('Issue'),
                            row.get('Industry'),
                            row.get('Comments'),
                            row.get('SpokespersonName')
                        )
                        
                        cursor.execute(insert_query, data)
                        successful_imports += 1
                    except Exception as e:
                        print(f"Error inserting row {index}: {e}")
                        continue
                
                connection.commit()
                print(f"Successfully imported {successful_imports} out of {len(df)} rows to staging table!")
                
                cursor.close()
                connection.close()
                
        except Exception as e:
            print(f"Error importing CSV to staging: {e}")

    def generate_embeddings_for_comments(self, batch_size=100):
        """
        Generate vector embeddings for Comments using MySQL HeatWave ML_EMBED_TABLE.
        This function processes the staging table and updates the Comments_embedding column.
        """
        connection = self.get_connection()
        if connection:
            try:
                cursor = connection.cursor()
                
                # First, check if there are any records without embeddings
                check_query = """
                SELECT COUNT(*) FROM articles 
                WHERE Comments IS NOT NULL 
                AND Comments != '' 
                AND Comments_embedding IS NULL
                """
                cursor.execute(check_query)
                total_records = cursor.fetchone()[0]
                
                if total_records == 0:
                    print("No records found that need embedding generation.")
                    cursor.close()
                    connection.close()
                    return
                
                print(f"Found {total_records} records that need embedding generation.")
                
                # Use MySQL HeatWave ML_EMBED_TABLE to generate embeddings in batches
                # This approach updates the table directly using ML_EMBED_TABLE
                embed_query = """
                UPDATE articles 
                SET Comments_embedding = (
                    SELECT sys.ML_EMBED_ROW(Comments, JSON_OBJECT("model_id", "all_minilm_l12_v2"))
                )
                WHERE Comments IS NOT NULL 
                AND Comments != '' 
                AND Comments_embedding IS NULL
                LIMIT %s
                """
                
                processed = 0
                while processed < total_records:
                    try:
                        cursor.execute(embed_query, (batch_size,))
                        affected_rows = cursor.rowcount
                        connection.commit()
                        
                        processed += affected_rows
                        print(f"Processed {processed}/{total_records} records...")
                        
                        if affected_rows < batch_size:
                            # No more records to process
                            break
                            
                    except Exception as e:
                        print(f"Error in batch processing: {e}")
                        connection.rollback()
                        break
                
                print(f"Successfully generated embeddings for {processed} records!")
                
                cursor.close()
                connection.close()
                
            except Error as e:
                print(f"Error generating embeddings: {e}")

    def generate_embeddings_alternative(self):
        """
        Alternative method using ML_EMBED_TABLE for batch processing.
        This creates a temporary result table with embeddings.
        """
        connection = self.get_connection()
        if connection:
            try:
                cursor = connection.cursor()
                
                # Create a temporary table to store results
                create_temp_query = """
                CREATE TEMPORARY TABLE temp_embeddings AS
                SELECT ArticleID, Comments,
                    sys.ML_EMBED_ROW(Comments, JSON_OBJECT("model_id", "all_minilm_l12_v2")) as embedding
                FROM articles 
                WHERE Comments IS NOT NULL 
                AND Comments != ''
                AND Comments_embedding IS NULL
                """
                
                cursor.execute(create_temp_query)
                print("Generated embeddings in temporary table...")
                
                # Update the main table with embeddings from temp table
                update_query = """
                UPDATE articles a
                INNER JOIN temp_embeddings t ON a.ArticleID = t.ArticleID
                SET a.Comments_embedding = t.embedding
                """
                
                cursor.execute(update_query)
                affected_rows = cursor.rowcount
                connection.commit()
                
                print(f"Successfully updated {affected_rows} records with embeddings!")
                
                # Drop temporary table
                cursor.execute("DROP TEMPORARY TABLE temp_embeddings")
                
                cursor.close()
                connection.close()
                
            except Error as e:
                print(f"Error in alternative embedding generation: {e}")

    def insert_article_data(self, article_data):
        """Insert article data into the articles table"""
        connection = self.get_connection()
        if connection:
            try:
                cursor = connection.cursor()
                
                insert_query = """
                INSERT INTO articles (
                    ArticleID, ArtDate, Month, Year, CompetName,
                    KPMGTotalImpact, DeloitteTotalImpact, EYTotalImpact, PwCTotalImpact,
                    Issue, Industry, Comments, SpokespersonName
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                cursor.execute(insert_query, article_data)
                connection.commit()
                print("Article data inserted successfully!")
                
                cursor.close()
                connection.close()
            except Error as e:
                print(f"Error inserting article data: {e}")
    
    def get_all_articles(self):
        """Retrieve all articles from the database"""
        query = "SELECT * FROM articles ORDER BY ArtDate DESC"
        return self.execute_query(query)
    
    def get_articles_by_year(self, year):
        """Retrieve articles by year"""
        query = "SELECT * FROM articles WHERE Year = %s ORDER BY ArtDate DESC"
        return self.execute_query(query, (year,))
    
    def get_articles_by_industry(self, industry):
        """Retrieve articles by industry"""
        query = "SELECT * FROM articles WHERE Industry = %s ORDER BY ArtDate DESC"
        return self.execute_query(query, (industry,))

    def get_articles_by_similarity(self, query_text, limit=10, similarity_threshold=0.8):
        """
        Find articles similar to the query text using vector similarity search.
        """
        connection = self.get_connection()
        if connection:
            try:
                cursor = connection.cursor()
                
                # First, generate embedding for the query text
                embed_query_text = """
                SELECT sys.ML_EMBED_ROW(%s, JSON_OBJECT("model_id", "all_minilm_l12_v2")) as query_embedding
                """
                cursor.execute(embed_query_text, (query_text,))
                query_embedding = cursor.fetchone()[0]
                
                # Find similar articles using DISTANCE function
                similarity_query = """
                SELECT ArticleID, ArtDate, CompetName, Issue, Industry, Comments, SpokespersonName,
                    (1 - DISTANCE(Comments_embedding, %s)) as similarity_score
                FROM articles 
                WHERE Comments_embedding IS NOT NULL
                AND (1 - DISTANCE(Comments_embedding, %s)) >= %s
                ORDER BY similarity_score DESC
                LIMIT %s
                """
                
                cursor.execute(similarity_query, (query_embedding, query_embedding, similarity_threshold, limit))
                results = cursor.fetchall()
                
                cursor.close()
                connection.close()
                
                return results
                
            except Error as e:
                print(f"Error in similarity search: {e}")
                return None
        
    def import_csv_to_articles(self, csv_file_path):
        """
        Import CSV data and generate embeddings.
        This is a wrapper function that imports to staging and then generates embeddings.
        """
        print("Step 1: Importing CSV to staging table...")
        self.import_csv_to_staging(csv_file_path)
        
        print("Step 2: Generating embeddings for comments...")
        self.generate_embeddings_for_comments()
        
        print("Import and embedding generation completed!")

if __name__ == "__main__":
    db = MySQLConnection()
    db.create_staging_table()
    db.create_article_table()  
    db.import_csv_to_articles('data/InputData_IndustryEconomics_Jul24-Jun25.xlsx') 