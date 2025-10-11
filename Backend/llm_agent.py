"""
llm_agent.py - Enhanced RAG-based LLM Agent for LungScope
Implements adaptive conversational medical assistant with context retrieval
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from database import DataRetrievalAgent

# ------------------------- Logging -------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/lungscope_chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ------------------------- Load Environment -------------------------
load_dotenv()

class LungScopeChatbot:
    """Adaptive RAG-based AI Doctor for respiratory health"""

    def __init__(self, temperature: float = 0.5, model_name: str = "gpt-4o-mini"):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        # Initialize LLM
        self.llm = ChatOpenAI(
            openai_api_key=self.openai_api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=400  # base limit, will adjust dynamically
        )

        # Retrieval and memory
        self.retrieval_agent = DataRetrievalAgent()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="input",
            output_key="output"
        )

        self.conversation_turns = {}
        self.system_prompt = self._build_system_prompt()
        self.qa_chain = None
        self._build_qa_chain()

        logger.info(f"LungScopeChatbot initialized with model: {model_name}")

    # ------------------------- PROMPT -------------------------
    def _build_system_prompt(self) -> str:
        """Doctor behavior system prompt"""
        return """You are Dr. AIRA (AI Respiratory Assistant), an AI respiratory health assistant.

CRITICAL BEHAVIOR RULES:
1. **ALWAYS** acknowledge patient's medical history in your FIRST response.
2. If patient has known chronic disease (COPD, Asthma, Chronic Bronchitis):
   - DO NOT request audio.
   - Focus on management advice and triggers.
3. If no known condition:
   - Gather 1–2 rounds of symptoms.
   - THEN ask for audio (Turn 3+).
4. Use short 2–4 sentence responses early, longer 5–8 sentence responses for final consultation.
5. Maintain empathy, professionalism, and clear actionable advice.

Detectable diseases (via audio): Bronchiectasis, Bronchiolitis, COPD, Pneumonia, URTI, Healthy lungs.
"""

    # ------------------------- BUILD QA CHAIN -------------------------
    def _build_qa_chain(self):
        """Initialize conversational retrieval chain"""
        retriever = self.retrieval_agent.vector_manager.get_retriever(search_kwargs={"k": 5})

        qa_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.system_prompt),
            HumanMessagePromptTemplate.from_template("""
Context from medical knowledge base:
{context}

Patient Info Summary:
{patient_context}

Audio Analysis:
{audio_analysis}

Patient Question: {question}
""")
        ])

        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            verbose=False
        )

    # ------------------------- FORMAT CONTEXT -------------------------
    def _format_patient_context(self, full_context: Dict[str, Any]) -> str:
        """
        Format patient context for LLM prompt (anonymized, structured)

        Args:
            full_context: Retrieved context from DataRetrievalAgent

        Returns:
            Formatted string for prompt injection
        """
        patient_data = full_context.get('patient_data')
        patient_history = full_context.get('patient_history', [])

        # Handle case where patient_data is None
        if not patient_data:
            logger.warning("No patient data found in context")
            return """Patient Profile: No patient data available

    This appears to be a new patient or the profile hasn't been loaded yet.
    Please ask the patient to provide their basic medical information."""

        # Build context string with patient data
        context_str = "Patient Profile (Anonymized):\n"
        context_str += f"- Age Range: {patient_data.get('age_range', 'Unknown')}\n"
        context_str += f"- Gender: {patient_data.get('gender', 'Unknown')}\n"
        context_str += f"- Smoking Status: {patient_data.get('smoking_status', 'Unknown')}\n"

        # Comorbidities
        comorbidities = []
        if patient_data.get('has_hypertension'):
            comorbidities.append('Hypertension')
        if patient_data.get('has_diabetes'):
            comorbidities.append('Diabetes')
        if patient_data.get('has_asthma_history'):
            comorbidities.append('History of Asthma')

        context_str += f"- Comorbidities: {', '.join(comorbidities) if comorbidities else 'None reported'}\n"
        context_str += f"- Current Medications: {patient_data.get('current_medications', 'None listed')}\n"
        context_str += f"- Known Allergies: {patient_data.get('allergies', 'None listed')}\n"

        # Recent medical history
        if patient_history:
            context_str += "\nRecent Medical History:\n"
            for idx, visit in enumerate(patient_history[:3], 1):
                context_str += f"{idx}. {visit.get('visit_date', 'Date unknown')}: {visit.get('diagnosis', 'N/A')} - {visit.get('symptoms', 'N/A')}\n"

        return context_str

    def _format_audio_analysis(self, audio_result: Dict[str, Any]) -> str:
        if not audio_result:
            return "No audio analysis yet."

        text = f"""
Disease Classification: {audio_result.get('disease', 'Unknown')}
Confidence: {audio_result.get('confidence', 0)*100:.1f}%
Severity: {audio_result.get('severity', 'Not specified')}
"""

        conf = audio_result.get('confidence', 0)
        if conf >= 0.9: text += "→ High reliability"
        elif conf >= 0.75: text += "→ Moderate reliability"
        else: text += "→ Low reliability, recommend doctor evaluation"

        return text

    # ------------------------- MAIN QUERY -------------------------
    def query(self, user_query: str, patient_id: str, audio_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Main query pipeline"""

        if patient_id not in self.conversation_turns:
            self.conversation_turns[patient_id] = 0

        self.conversation_turns[patient_id] += 1
        turn = self.conversation_turns[patient_id]

        # Conversation phase logic
        if turn <= 2:
            phase = "intro"
        elif turn == 3:
            phase = "analysis"
        else:
            phase = "consultation"

        disease_class = audio_result.get("disease") if audio_result else "Unknown"

        full_context = self.retrieval_agent.retrieve_full_context(
            patient_id=patient_id,
            disease_classification=disease_class,
            user_query=user_query,
            audio_result=audio_result
        )

        patient_context = self._format_patient_context(full_context)
        audio_analysis = self._format_audio_analysis(audio_result)

        # Query LLM
        response = self._query_with_context(
            user_query, patient_context, audio_analysis,
            full_context.get("disease_knowledge", []),
            turn, bool(audio_result), phase
        )

        follow_up = self._generate_follow_up(disease_class, audio_result, turn)
        confidence = self._calculate_confidence(audio_result, full_context)

        structured = {
            "response": response,
            "confidence": float(confidence),
            "disease_classification": disease_class,
            "follow_up": follow_up,
            "turn_count": turn,
            "should_request_audio": turn >= 3 and not audio_result,
            "timestamp": datetime.now().isoformat()
        }

        self.retrieval_agent.log_interaction(patient_id, user_query, response, audio_result)
        return structured

    # ------------------------- QUERY WITH CONTEXT -------------------------
    def _query_with_context(
        self, question: str, patient_context: str, audio_analysis: str,
        disease_knowledge: List[Dict], turn: int, has_audio: bool, phase: str
    ) -> str:

        # Knowledge context
        knowledge_context = "\n".join(
            [f"Source {i+1}: {doc.get('content', '')[:150]}..." for i, doc in enumerate(disease_knowledge[:3])]
        ) or "No specific disease knowledge found."

        # CRITICAL: Control tone based on turn and audio status
        if turn == 1:
            # TURN 1: Acknowledge history
            tone = """**MANDATORY FIRST RESPONSE:**
    You MUST start by acknowledging patient's medical profile.
    If comorbidities exist: "Hello! I see from your profile that you have [conditions]..."
    If smoking history: "As a [former/current] smoker..."
    If no history: "Hello! Welcome to your first consultation..."
    Then address their question in 2-3 sentences."""
            self.llm.max_tokens = 500
            prompt_patient_context = f"===== PATIENT PROFILE =====\n{patient_context}\n=========================="

        elif turn == 2:
            # TURN 2: Gather more symptoms
            tone = """Ask 2-3 follow-up questions about their symptoms:
    - Duration and severity
    - Associated symptoms (fever, chest pain, wheezing)
    - Triggers or recent exposures
    Keep response brief (3-4 sentences)."""
            self.llm.max_tokens = 400
            prompt_patient_context = f"Patient Profile:\n{patient_context}"

        elif turn == 3 and not has_audio:
            # TURN 3: FORCE AUDIO REQUEST
            tone = """**CRITICAL - REQUEST AUDIO RECORDING:**

    Based on the symptoms described (shortness of breath, coughing, etc.), you now have enough information.

    Your response MUST:
    1. Summarize their symptoms (1-2 sentences)
    2. Explain possible causes (2 sentences): "Based on your symptoms of [list], possible causes include acute bronchitis, upper respiratory infection, or asthma exacerbation."
    3. **REQUEST AUDIO**: "To provide a more accurate assessment, I'd like to analyze your lung sounds. Could you please record and upload a 10-15 second audio of your breathing? This will help me determine if there's any concerning wheezing, crackling, or other abnormal sounds."

    End with: "Click the attachment button to upload your lung sound recording."

    **DO NOT ask more symptom questions. MUST request audio."""
            self.llm.max_tokens = 500
            prompt_patient_context = f"Patient Profile:\n{patient_context}"

        elif has_audio and turn >= 4:
            # FINAL CONSULTATION: Provide comprehensive advice
            tone = """**FINAL CONSULTATION WITH AUDIO RESULTS:**

    Structure your response:
    1. **Audio Result** (1 sentence): "Your lung sound analysis shows [result] with [X]% confidence."

    2. **Symptom Analysis** (2-3 sentences):
    - If Healthy but symptoms exist: "While your lungs sound healthy, your symptoms suggest an acute condition like bronchitis or viral infection. Audio can be normal if inflammation is mild."
    - If disease detected: "Your symptoms combined with audio analysis suggest [diagnosis]."

    3. **Medical Advice** (2-3 sentences):
    - Most likely diagnosis
    - When to seek immediate care
    - Expected recovery time

    4. **Do's and Don'ts**:
    **DO:**
    - Rest and hydrate (8+ glasses/day)
    - Steam inhalation 2-3x daily
    - Monitor symptoms

    **DON'T:**
    - Smoke or expose to irritants
    - Ignore worsening symptoms
    - Self-medicate antibiotics

    5. **Follow-up**: "Seek care if symptoms persist beyond 7-10 days or worsen."

    **DO NOT mention medical history.**"""
            self.llm.max_tokens = 800
            prompt_patient_context = "Focus on current symptoms and audio results."

        else:
            # DEFAULT: Continue gathering info
            tone = "Ask follow-up questions about symptoms. Keep brief."
            self.llm.max_tokens = 400
            prompt_patient_context = f"Patient Profile:\n{patient_context}"

        # Build prompt
        full_prompt = f"""
    {self.system_prompt}

    Phase: {phase.upper()} | Turn: {turn}
    Instructions: {tone}

    {prompt_patient_context}

    Audio Analysis:
    {audio_analysis}

    Medical Knowledge:
    {knowledge_context}

    Patient's Question: {question}

    Your Response:
    """

        response = self.llm.invoke(full_prompt)
        response_text = response.content.strip()

        if turn == 3 and not has_audio:
            logger.info(f"Turn 3: MUST request audio. Response: {response_text[:100]}...")

        # Save to memory
        self.memory.save_context({"input": question}, {"output": response_text})

        return response_text



    # ------------------------- FOLLOW-UP -------------------------
    def _generate_follow_up(self, disease: str, audio_result: Optional[Dict], turn: int) -> str:
        if audio_result:
            c = audio_result.get("confidence", 0.0)
            if c > 0.8:
                return "Schedule a follow-up with a pulmonologist if symptoms persist or worsen."
            else:
                return "Audio analysis inconclusive; consider in-person evaluation."

        if turn >= 3:
            return "Please record your lung sounds for further analysis."
        elif turn == 1:
            return "Please describe when symptoms began and any known triggers."
        else:
            return "Could you share if you experience chest pain or coughing with phlegm?"

    # ------------------------- CONFIDENCE -------------------------
    def _calculate_confidence(self, audio_result: Optional[Dict], full_context: Dict) -> float:
        conf = 0.5
        if audio_result: conf = audio_result.get("confidence", 0.5) * 0.6
        if full_context.get("patient_data"): conf += 0.15
        if len(full_context.get("disease_knowledge", [])) >= 3: conf += 0.15
        if full_context.get("patient_history"): conf += 0.10
        return min(conf, 1.0)

    # ------------------------- UTILITIES -------------------------
    def reset_memory(self):
        self.memory.clear()
        logger.info("Conversation memory reset")

    def cleanup(self):
        self.retrieval_agent.cleanup()
        logger.info("LungScopeChatbot cleanup complete")
