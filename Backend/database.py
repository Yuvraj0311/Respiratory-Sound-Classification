"""
database.py - Data Retrieval Agent for AIRA (FIXED VERSION)
Handles PostgreSQL patient data and ChromaDB vector retrieval
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import bcrypt

LOGS_DIR = 'logs'
os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'lungscope_audit.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()


class DatabaseManager:
    """Manages PostgreSQL connections for patient data retrieval"""
    
    def __init__(self):
        """Initialize and validate database connection parameters from .env"""
        
        self.db_user = os.getenv('POSTGRES_USER')
        self.db_password = os.getenv('POSTGRES_PASSWORD')
        self.db_name = os.getenv('POSTGRES_DB')
        
        if not all([self.db_user, self.db_password, self.db_name]):
            missing_vars = [
                var for var, val in 
                [("POSTGRES_USER", self.db_user), ("POSTGRES_PASSWORD", self.db_password), ("POSTGRES_DB", self.db_name)] 
                if not val
            ]
            error_message = f"FATAL ERROR: Missing required environment variables: {', '.join(missing_vars)}. Please ensure your .env file is in the 'Backend' directory and contains these keys."
            logger.critical(error_message)
            raise ValueError(error_message)

        self.db_config = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'user': self.db_user,
            'password': self.db_password,
            'database': self.db_name,
            'port': int(os.getenv('POSTGRES_PORT', 5432))
        }
        self.conn = None
        logger.info("DatabaseManager initialized")

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using bcrypt"""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            logger.info("Successfully connected to PostgreSQL database")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def disconnect(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def get_patient_data(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve anonymized patient medical history from database"""
        try:
            if not self.conn:
                self.connect()
            
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            
            query = """
            SELECT 
                patient_id,
                email,
                username,
                full_name,
                age_range,
                gender,
                smoking_status,
                has_hypertension,
                has_diabetes,
                has_asthma_history,
                previous_respiratory_infections,
                current_medications,
                allergies,
                last_consultation_date,
                avatar
            FROM patients
            WHERE patient_id = %s
            """
            
            cursor.execute(query, (patient_id,))
            result = cursor.fetchone()
            cursor.close()
            
            if result:
                patient_data = dict(result)
                logger.info(f"Patient data retrieved for ID: {patient_id[:6]}***")
                return patient_data
            else:
                logger.warning(f"No patient data found for ID: {patient_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving patient data: {e}")
            return None
    
    def get_patient_history(self, patient_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve recent medical history for contextualization"""
        try:
            if not self.conn:
                self.connect()
            
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            
            query = """
            SELECT 
                visit_date,
                diagnosis,
                symptoms,
                treatment_prescribed,
                follow_up_required
            FROM medical_history
            WHERE patient_id = %s
            ORDER BY visit_date DESC
            LIMIT %s
            """
            
            cursor.execute(query, (patient_id, limit))
            results = cursor.fetchall()
            cursor.close()
            
            history = [dict(row) for row in results]
            logger.info(f"Retrieved {len(history)} historical records for patient")
            return history
            
        except Exception as e:
            logger.error(f"Error retrieving patient history: {e}")
            return []
    
    def log_query(
        self,
        patient_id: str,
        query_text: str,
        response: str,
        audio_result: Optional[Dict] = None,
        xray_result: Optional[Dict] = None
    ):
        """Log all queries and responses for HIPAA audit trail"""
        try:
            if not self.conn:
                self.connect()
            
            cursor = self.conn.cursor()
            
            query = """
            INSERT INTO audit_log 
                (patient_id, query_timestamp, query_text, response_text, 
                 audio_analysis_result, created_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            """

            tool_results = {}
            if audio_result:
                tool_results["audio"] = audio_result
            if xray_result:
                tool_results["xray"] = xray_result
            
            cursor.execute(query, (
                patient_id,
                datetime.now(),
                query_text,
                response,
                json.dumps(tool_results) if tool_results else None,
                datetime.now()
            ))
            
            self.conn.commit()
            cursor.close()
            logger.info(f"Audit log created for patient: {patient_id[:6]}***")
            
        except Exception as e:
            logger.error(f"Error logging query to audit trail: {e}")
    
    def get_patient_for_auth(self, username_or_email: str) -> Optional[Dict[str, Any]]:
        """Get patient for authentication by username or email"""
        try:
            if not self.conn:
                self.connect()
            
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            
            query = """
                SELECT patient_id, username, email, password_hash, auth_provider 
                FROM patients 
                WHERE username = %s OR email = %s
            """
            
            cursor.execute(query, (username_or_email, username_or_email))
            result = cursor.fetchone()
            cursor.close()
            
            if result:
                return dict(result)
            else:
                logger.warning(f"Authentication attempt for non-existent user: {username_or_email}")
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving patient for auth: {e}")
            return None
    
    def create_patient(self, patient_data: Dict[str, Any]) -> bool:
        """Inserts a new patient record into the database WITH hashed password"""
        try:
            if not self.conn:
                self.connect()
            
            cursor = self.conn.cursor()
            
            # Hash password if provided (for local auth)
            password_hash = None
            if patient_data.get('password'):
                password_hash = self.hash_password(patient_data['password'])
            
            query = """
            INSERT INTO patients (
                patient_id, email, username, full_name, password_hash, age_range, 
                gender, smoking_status, has_hypertension, has_diabetes, 
                has_asthma_history, previous_respiratory_infections, 
                current_medications, allergies, avatar, auth_provider
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            """
            
            data_tuple = (
                patient_data['patient_id'],
                patient_data.get('email'),
                patient_data.get('username'),
                patient_data.get('full_name'),
                password_hash,  # Use hashed password
                patient_data.get('age_range'),
                patient_data.get('gender'),
                patient_data.get('smoking_status'),
                patient_data.get('has_hypertension', False),
                patient_data.get('has_diabetes', False),
                patient_data.get('has_asthma_history', False),
                patient_data.get('previous_respiratory_infections', 0),
                patient_data.get('current_medications', ''),
                patient_data.get('allergies', ''),
                patient_data.get('avatar', None),
                patient_data.get('auth_provider', 'local')
            )
            
            cursor.execute(query, data_tuple)
            self.conn.commit()
            cursor.close()
            
            logger.info(f"Successfully registered new patient: {patient_data['username']}")
            return True
        
        except psycopg2.IntegrityError as e:
            logger.warning(f"Registration failed: Patient already exists. Error: {e}")
            self.conn.rollback()
            return False
        except Exception as e:
            logger.error(f"Error creating patient: {e}")
            self.conn.rollback()
            return False
    
    def update_patient(self, patient_id: str, update_data: Dict[str, Any]) -> bool:
        """Updates an existing patient record in the database"""
        try:
            if not self.conn:
                self.connect()
            
            cursor = self.conn.cursor()
            
            set_clauses = []
            values = []
            
            allowed_fields = [
                'email', 'full_name', 'age_range', 'gender', 'smoking_status', 
                'has_hypertension', 'has_diabetes', 'has_asthma_history', 
                'previous_respiratory_infections', 'current_medications', 
                'allergies', 'last_consultation_date', 'avatar'
            ]
            
            for field in allowed_fields:
                if field in update_data:
                    set_clauses.append(f"{field} = %s")
                    values.append(update_data[field])
            
            if 'password' in update_data and update_data['password']:
                set_clauses.append("password_hash = %s")
                values.append(self.hash_password(update_data['password']))
            
            if not set_clauses:
                logger.warning("No valid fields provided for update")
                return False
            
            set_clauses.append("updated_at = %s")
            values.append(datetime.now())
            values.append(patient_id)
            
            query = f"""
            UPDATE patients 
            SET {', '.join(set_clauses)}
            WHERE patient_id = %s
            """
            
            cursor.execute(query, values)
            self.conn.commit()
            rows_affected = cursor.rowcount
            cursor.close()
            
            if rows_affected > 0:
                logger.info(f"Successfully updated patient: {patient_id}")
                return True
            else:
                logger.warning(f"No patient found with ID: {patient_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating patient: {e}")
            self.conn.rollback()
            return False


class VectorDBManager:
    """Manages ChromaDB vector database for medical knowledge retrieval"""
    
    def __init__(self, persist_directory: str = "./vector_db/chroma_db"):
        """Initialize ChromaDB client with OpenAI embeddings"""
        self.persist_directory = persist_directory
        self.collection_name = "medical_knowledge"
        
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            model="text-embedding-3-small"
        )
        
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        self.vectorstore = None
        self._initialize_vectorstore()
        
        logger.info(f"VectorDBManager initialized with directory: {persist_directory}")
    
    def _initialize_vectorstore(self):
        """Initialize or load existing vectorstore"""
        try:
            self.vectorstore = Chroma(
                client=self.client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            logger.info("Vectorstore initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing vectorstore: {e}")
            raise
    
    def retrieve_disease_context(
        self, 
        disease_name: str, 
        query: Optional[str] = None,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant medical knowledge for a specific disease"""
        try:
            search_query = f"{disease_name}"
            if query:
                search_query += f" {query}"
            
            filter_dict = {"disease": disease_name}
            
            docs = self.vectorstore.similarity_search(
                query=search_query,
                k=k,
                filter=filter_dict
            )
            
            results = []
            for doc in docs:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "source": doc.metadata.get("source", "Unknown")
                })
            
            logger.info(f"Retrieved {len(results)} documents for disease: {disease_name}")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving disease context: {e}")
            return []
    
    def retrieve_comorbidity_context(
        self, 
        disease: str, 
        comorbidities: List[str],
        k: int = 3
    ) -> List[Dict[str, Any]]:
        """Retrieve context about disease interactions with comorbidities"""
        try:
            results = []
            
            for comorbidity in comorbidities:
                search_query = f"{disease} with {comorbidity} comorbidity treatment precautions"
                
                docs = self.vectorstore.similarity_search(
                    query=search_query,
                    k=k
                )
                
                for doc in docs:
                    results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "comorbidity": comorbidity
                    })
            
            logger.info(f"Retrieved {len(results)} comorbidity context documents")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving comorbidity context: {e}")
            return []
    
    def get_retriever(self, search_kwargs: Optional[Dict] = None):
        """Get LangChain retriever for conversational chains"""
        if search_kwargs is None:
            search_kwargs = {"k": 5}
        
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)


class DataRetrievalAgent:
    """Unified agent that coordinates patient DB and vector DB retrieval"""
    
    def __init__(self):
        """Initialize both database managers"""
        self.db_manager = DatabaseManager()
        self.vector_manager = VectorDBManager()
        logger.info("DataRetrievalAgent initialized")
    
    def retrieve_full_context(
        self,
        patient_id: str,
        disease_classification: str,
        user_query: Optional[str] = None,
        audio_result: Optional[Dict] = None,
        xray_result: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Retrieve comprehensive context for LLM agent"""
        try:
            patient_data = self.db_manager.get_patient_data(patient_id)
            patient_history = self.db_manager.get_patient_history(patient_id, limit=3)
            
            disease_context = self.vector_manager.retrieve_disease_context(
                disease_name=disease_classification,
                query=user_query,
                k=5
            )
            
            comorbidity_context = []
            if patient_data:
                comorbidities = []
                if patient_data.get('has_hypertension'):
                    comorbidities.append('hypertension')
                if patient_data.get('has_diabetes'):
                    comorbidities.append('diabetes')
                
                if comorbidities:
                    comorbidity_context = self.vector_manager.retrieve_comorbidity_context(
                        disease=disease_classification,
                        comorbidities=comorbidities,
                        k=2
                    )
            
            full_context = {
                "patient_data": patient_data,
                "patient_history": patient_history,
                "disease_classification": disease_classification,
                "audio_analysis": audio_result,
                "xray_analysis": xray_result,
                "disease_knowledge": disease_context,
                "comorbidity_knowledge": comorbidity_context,
                "retrieved_at": datetime.now().isoformat()
            }
            
            logger.info(f"Full context retrieved for patient: {patient_id[:6]}***")
            return full_context
            
        except Exception as e:
            logger.error(f"Error retrieving full context: {e}")
            return {}
    
    def log_interaction(
        self,
        patient_id: str,
        query: str,
        response: str,
        audio_result: Optional[Dict] = None,
        xray_result: Optional[Dict] = None
    ):
        """Log interaction to audit trail"""
        self.db_manager.log_query(patient_id, query, response, audio_result, xray_result)
    
    def cleanup(self):
        """Clean up database connections"""
        self.db_manager.disconnect()
        logger.info("DataRetrievalAgent cleanup complete")


if __name__ == "__main__":
    agent = DataRetrievalAgent()
    
    context = agent.retrieve_full_context(
        patient_id="PATIENT_12345",
        disease_classification="COPD",
        user_query="What should I do about my breathing difficulty?",
        audio_result={
            "disease": "COPD",
            "confidence": 0.9759,
            "severity": "moderate"
        }
    )
    
    print("Retrieved Context:")
    print(f"Patient Data: {context.get('patient_data')}")
    print(f"Disease Knowledge Documents: {len(context.get('disease_knowledge', []))}")
    
    agent.cleanup()
