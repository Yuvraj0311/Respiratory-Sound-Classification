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
from diagnostic_tool_selector import DiagnosticToolDecision, DeterministicDiagnosticToolSelector

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

    def __init__(self, temperature: float = 0.5, model_name: str = "gpt-4.1-nano"):
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
        self.symptom_history = {}
        self.tool_selector = DeterministicDiagnosticToolSelector()
        self.system_prompt = self._build_system_prompt()
        self.qa_chain = None
        self._build_qa_chain()

        logger.info(f"LungScopeChatbot initialized with model: {model_name}")

    # ------------------------- PROMPT -------------------------
    def _build_system_prompt(self) -> str:
        """Doctor behavior system prompt"""
        return """You are Dr. AIRA (AI Respiratory Assistant), an AI respiratory health assistant.

CRITICAL BEHAVIOR RULES:
1. Acknowledge patient medical history in the first medical response, not for casual greetings or non-medical small talk.
2. If patient has known chronic disease (COPD, Asthma, Chronic Bronchitis):
   - DO NOT request unnecessary diagnostics.
   - Focus on management advice and triggers.
3. If no known condition:
   - Gather 1-2 rounds of symptoms.
   - THEN ask for the deterministic tool selection based on symptoms. Lung sounds are favored for airway/noisy-breathing patterns; chest X-ray is favored for fever + cough + chest pain/worsening lower-respiratory patterns.
4. Use short 2-4 sentence responses early, longer 5-8 sentence responses for final consultation.
5. Maintain empathy, professionalism, and clear actionable advice.
6. When a Deterministic Tool Decision is provided, follow it exactly for upload requests.

Detectable diseases (via audio): Bronchiectasis, Bronchiolitis, COPD, Pneumonia, URTI, Healthy lungs.
Detectable X-ray findings (via chest X-ray image): Lung Opacity, Normal, Viral Pneumonia.
Diagnostic tool outputs are screening support, not a confirmed diagnosis; advise clinician/radiologist review for abnormal or low-confidence results.
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

X-ray Analysis:
{xray_analysis}

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
        if conf >= 0.9: text += "High reliability"
        elif conf >= 0.75: text += "Moderate reliability"
        else: text += "Low reliability, recommend doctor evaluation"

        return text

    def _format_xray_analysis(self, xray_result: Dict[str, Any]) -> str:
        if not xray_result:
            return "No chest X-ray analysis yet."

        text = f"""
Chest X-ray Finding: {xray_result.get('finding', 'Unknown')}
Confidence: {xray_result.get('confidence', 0)*100:.1f}%
Severity: {xray_result.get('severity', 'Not specified')}
Clinical Note: {xray_result.get('clinical_note', 'Screening result only; correlate clinically.')}
"""

        top_predictions = xray_result.get("top_predictions") or []
        if top_predictions:
            top_text = ", ".join(
                f"{item.get('finding', 'Unknown')} {item.get('confidence', 0)*100:.1f}%"
                for item in top_predictions[:3]
            )
            text += f"Top Predictions: {top_text}\n"

        conf = xray_result.get('confidence', 0)
        if conf >= 0.9:
            text += "Reliability: High model confidence; still requires clinical/radiology review."
        elif conf >= 0.75:
            text += "Reliability: Moderate model confidence; recommend clinician/radiologist review."
        else:
            text += "Reliability: Low confidence; do not rely on this result alone."

        return text

    def _format_tool_decision(self, tool_decision: DiagnosticToolDecision) -> str:
        questions = "\n".join(
            f"- {question}" for question in tool_decision.detail_questions[:3]
        ) or "- Ask about duration, severity, associated symptoms, and red flags."

        return f"""
Deterministic Tool Decision:
- Recommended Action: {tool_decision.action}
- Symptom Focus: {tool_decision.symptom_focus}
- Reason: {tool_decision.reason}
- Audio Score: {tool_decision.audio_score}
- X-ray Score: {tool_decision.xray_score}
- Relevant Detail Questions:
{questions}
"""

    def _select_primary_condition(
        self,
        audio_result: Optional[Dict[str, Any]],
        xray_result: Optional[Dict[str, Any]]
    ) -> str:
        if audio_result and audio_result.get("disease"):
            return audio_result.get("disease")
        if xray_result:
            return (
                xray_result.get("knowledge_disease")
                or xray_result.get("finding")
                or "Unknown"
            )
        return "Unknown"

    # ------------------------- MAIN QUERY -------------------------
    def query(
        self,
        user_query: str,
        patient_id: str,
        conversation_id: Optional[str] = None,
        recent_messages: Optional[List[str]] = None,
        audio_result: Optional[Dict[str, Any]] = None,
        xray_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Main query pipeline"""

        conversation_key = f"{patient_id}:{conversation_id or 'default'}"

        if conversation_key not in self.conversation_turns:
            self.conversation_turns[conversation_key] = 0

        self.conversation_turns[conversation_key] += 1
        turn = self.conversation_turns[conversation_key]

        patient_history = self.symptom_history.setdefault(conversation_key, [])
        incoming_messages = [
            message for message in (recent_messages or [])
            if isinstance(message, str) and message.strip()
        ]
        if user_query and user_query not in incoming_messages:
            incoming_messages.append(user_query)
        if incoming_messages:
            patient_history.extend(incoming_messages)
        elif user_query:
            patient_history.append(user_query)
        deduped_history = list(dict.fromkeys(patient_history))
        self.symptom_history[conversation_key] = deduped_history[-10:]
        recent_symptoms = self.symptom_history[conversation_key][-6:]
        tool_decision = self.tool_selector.select(
            recent_symptoms,
            has_audio_result=bool(audio_result),
            has_xray_result=bool(xray_result)
        )
        logger.info(
            "Tool decision for patient %s conversation %s: action=%s, audio_score=%s, xray_score=%s, has_audio=%s, has_xray=%s",
            patient_id[:6],
            (conversation_id or "default")[:8],
            tool_decision.action,
            tool_decision.audio_score,
            tool_decision.xray_score,
            bool(audio_result),
            bool(xray_result),
        )

        # Conversation phase logic
        if turn <= 2:
            phase = "intro"
        elif turn == 3:
            phase = "analysis"
        else:
            phase = "consultation"

        disease_class = self._select_primary_condition(audio_result, xray_result)

        full_context = self.retrieval_agent.retrieve_full_context(
            patient_id=patient_id,
            disease_classification=disease_class,
            user_query=user_query,
            audio_result=audio_result,
            xray_result=xray_result
        )

        patient_context = self._format_patient_context(full_context)
        audio_analysis = self._format_audio_analysis(audio_result)
        xray_analysis = self._format_xray_analysis(xray_result)
        tool_decision_context = self._format_tool_decision(tool_decision)

        # Query LLM
        response = self._query_with_context(
            user_query, patient_context, audio_analysis, xray_analysis, tool_decision_context, tool_decision,
            recent_symptoms,
            full_context.get("disease_knowledge", []),
            turn, bool(audio_result), bool(xray_result), phase
        )

        follow_up = self._generate_follow_up(disease_class, audio_result, xray_result, turn, tool_decision)
        confidence = self._calculate_confidence(audio_result, xray_result, full_context)
        should_prompt_for_tool = (
            turn >= 3
            and not audio_result
            and not xray_result
            and tool_decision.action != "urgent_care"
        )

        structured = {
            "response": response,
            "confidence": float(confidence),
            "disease_classification": disease_class,
            "xray_classification": xray_result.get("finding") if xray_result else None,
            "follow_up": follow_up,
            "turn_count": turn,
            "should_request_audio": should_prompt_for_tool and tool_decision.should_request_audio,
            "should_request_xray": should_prompt_for_tool and tool_decision.should_request_xray,
            "recommended_tool": tool_decision.action,
            "tool_selection_reason": tool_decision.reason,
            "diagnostic_tool_decision": tool_decision.to_dict(),
            "timestamp": datetime.now().isoformat()
        }

        self.retrieval_agent.log_interaction(patient_id, user_query, response, audio_result, xray_result)
        return structured

    # ------------------------- QUERY WITH CONTEXT -------------------------
    def _query_with_context(
        self, question: str, patient_context: str, audio_analysis: str, xray_analysis: str,
        tool_decision_context: str, tool_decision: DiagnosticToolDecision,
        recent_symptoms: List[str],
        disease_knowledge: List[Dict], turn: int, has_audio: bool, has_xray: bool, phase: str
    ) -> str:

        # Knowledge context
        knowledge_context = "\n".join(
            [f"Source {i+1}: {doc.get('content', '')[:150]}..." for i, doc in enumerate(disease_knowledge[:3])]
        ) or "No specific disease knowledge found."

        has_diagnostic_result = has_audio or has_xray
        recent_conversation = "\n".join(
            f"- {message}" for message in recent_symptoms[-6:]
        ) or "- No recent user symptom messages available."
        detail_questions_text = "\n".join(
            f"    - {item}" for item in tool_decision.detail_questions[:3]
        ) or "    - Ask about duration, severity, associated symptoms, and red flags."

        # CRITICAL: Control tone based on turn and diagnostic tool status
        if tool_decision.action == "conversation":
            tone = """**NORMAL CONVERSATION RESPONSE:**
    The user has not described symptoms or asked a medical question yet.
    Reply naturally in 1-2 short sentences.
    Do not mention patient medical history.
    Do not ask for audio, X-ray, or diagnostic uploads.
    Invite them to share symptoms or a health question if they want help."""
            self.llm.max_tokens = 180
            prompt_patient_context = "Do not use patient profile details for this casual response."

        elif has_diagnostic_result:
            # Diagnostic results must win over turn count. This keeps follow-up
            # uploads coherent even if uvicorn reload resets in-memory turns.
            tone = """**CONSULTATION WITH DIAGNOSTIC TOOL RESULTS:**

    Structure your response:
    1. Tool result summary:
    - If audio is present, state the lung sound result and confidence.
    - If X-ray is present, state the chest X-ray finding and confidence.
    - If an X-ray result is abnormal or low-confidence, recommend clinician/radiologist review.

    2. Symptom analysis:
    - Combine the user's symptoms with available tool outputs.
    - Do not overstate certainty. Use "suggests", "may indicate", or "screening output" when appropriate.

    3. Medical advice:
    - Most likely next steps.
    - When to seek urgent care, especially severe breathlessness, chest pain, blue lips, confusion, or oxygen saturation below 92%.
    - Expected follow-up path.

    4. Do's and Don'ts:
    **DO:**
    - Rest, hydrate, and monitor symptoms
    - Follow prescribed medicines
    - Arrange clinical review for abnormal X-ray findings

    **DON'T:**
    - Smoke or expose yourself to respiratory irritants
    - Ignore worsening symptoms
    - Self-medicate antibiotics or steroids without a clinician

    Do not ask the patient to upload the same diagnostic file again.
    Do not restart the conversation or repeat the first-response profile greeting."""
            self.llm.max_tokens = 850
            prompt_patient_context = "Focus on current symptoms and diagnostic tool results."

        elif tool_decision.action == "urgent_care":
            tone = f"""**URGENT SAFETY RESPONSE:**
    The deterministic tool selector found red flags. Do not request audio or X-ray upload first.
    Tell the patient to seek urgent medical care now or call emergency services if symptoms are severe.
    Ask only the most important safety question if needed:
{detail_questions_text}
    Keep the response direct and calm."""
            self.llm.max_tokens = 500
            prompt_patient_context = f"Patient Profile:\n{patient_context}"

        elif turn == 1:
            # TURN 1: Acknowledge history
            tone = f"""**MANDATORY FIRST RESPONSE WITH DETERMINISTIC DETAIL GATHERING:**
    You MUST start by acknowledging patient's medical profile.
    If comorbidities exist: "Hello! I see from your profile that you have [conditions]..."
    If smoking history: "As a [former/current] smoker..."
    If no history: "Hello! Welcome to your first consultation..."
    Then address their question in 2-3 sentences.
    If a diagnostic tool result is present, include the result briefly and state that it is screening support, not a confirmed diagnosis.
    Ask these relevant detail questions based on the deterministic tool selector:
{detail_questions_text}"""
            self.llm.max_tokens = 650
            prompt_patient_context = f"===== PATIENT PROFILE =====\n{patient_context}\n=========================="

        elif turn == 2:
            # TURN 2: Gather more symptoms
            tone = f"""Ask 2-3 follow-up questions chosen from this deterministic list:
{detail_questions_text}
    Keep response brief (3-4 sentences).
    Do not request a diagnostic upload yet unless the patient explicitly says they already have the file ready."""
            self.llm.max_tokens = 400
            prompt_patient_context = f"Patient Profile:\n{patient_context}"

        elif turn >= 3 and not has_diagnostic_result:
            # TURN 3: Request the appropriate diagnostic tool upload
            if tool_decision.action == "audio":
                upload_request = """Request a lung sound recording only.
    Say: "Please upload a 10-15 second lung sound or breathing audio recording using the attachment button."
    Do not request an X-ray image unless the patient says they already have one."""
            elif tool_decision.action == "xray":
                upload_request = """Request a chest X-ray image because the symptom pattern is imaging-relevant.
    Say: "Your symptoms make a chest X-ray useful for screening support. If you already have a chest X-ray image, please upload it using the attachment button."
    If they do not have an image, advise clinical review to decide whether chest imaging is needed."""
            elif tool_decision.action == "both":
                upload_request = """Request both relevant tools, while allowing one upload now.
    Say: "Please upload either a lung sound recording or an existing chest X-ray image; if you have both, start with either one and send the other next."
    Explain that audio helps with airway sounds and X-ray helps with opacity/pneumonia screening."""
            else:
                upload_request = """Symptoms are not strongly routed to one tool.
    Ask for either available option:
    "You can upload a lung sound recording, or if you already have a chest X-ray image, upload that instead."
    Let the patient choose whichever file they can provide."""

            tone = f"""**CRITICAL - REQUEST THE DETERMINISTIC DIAGNOSTIC TOOL:**

    Based on the symptoms described (shortness of breath, coughing, etc.), you now have enough information.

    Your response MUST:
    1. Summarize their symptoms (1-2 sentences)
    2. Explain possible causes briefly without diagnosing.
    3. Follow this deterministic upload request exactly:
{upload_request}

    **DO NOT ask more symptom questions. MUST request the selected upload path.**"""
            self.llm.max_tokens = 500
            prompt_patient_context = f"Patient Profile:\n{patient_context}"

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

    Chest X-ray Analysis:
    {xray_analysis}

    Diagnostic Tool Selection:
    {tool_decision_context}

    Recent User Symptom Messages:
    {recent_conversation}

    Medical Knowledge:
    {knowledge_context}

    Patient's Question: {question}

    Your Response:
    """

        response = self.llm.invoke(full_prompt)
        response_text = response.content.strip()

        if turn == 3 and not has_diagnostic_result:
            logger.info(f"Turn 3: requested diagnostic upload. Response: {response_text[:100]}...")

        # Save to memory
        self.memory.save_context({"input": question}, {"output": response_text})

        return response_text



    # ------------------------- FOLLOW-UP -------------------------
    def _generate_follow_up(
        self,
        disease: str,
        audio_result: Optional[Dict],
        xray_result: Optional[Dict],
        turn: int,
        tool_decision: DiagnosticToolDecision
    ) -> str:
        if tool_decision.action == "conversation":
            return "Share any symptoms or health question when you're ready."

        if audio_result and xray_result:
            return "Please review these combined lung sound and chest X-ray screening results with a clinician if symptoms persist or worsen."

        if xray_result:
            c = xray_result.get("confidence", 0.0)
            finding = xray_result.get("finding", "X-ray finding")
            if finding != "Normal":
                return "Please review this X-ray result with a clinician or radiologist, especially if symptoms persist or worsen."
            if c > 0.8:
                return "If symptoms continue despite a normal X-ray screening result, arrange a clinical evaluation."
            return "X-ray analysis was inconclusive; consider a clinician or radiologist review."

        if audio_result:
            c = audio_result.get("confidence", 0.0)
            if c > 0.8:
                return "Schedule a follow-up with a pulmonologist if symptoms persist or worsen."
            else:
                return "Audio analysis inconclusive; consider in-person evaluation."

        if tool_decision.action == "urgent_care":
            return "Seek urgent medical care now if red-flag symptoms are present."

        if turn >= 3:
            if tool_decision.action == "audio":
                return "Please upload a 10-15 second lung sound recording for respiratory sound analysis."
            if tool_decision.action == "xray":
                return "A chest X-ray may be useful based on your symptoms; upload one if available, otherwise arrange clinical review to decide whether imaging is needed."
            if tool_decision.action == "both":
                return "Please upload either a lung sound recording or an existing chest X-ray image; both can help from different angles."
            return "Please upload either a lung sound recording or an existing chest X-ray image, whichever you can provide."
        elif turn == 1:
            return "Please describe when symptoms began and any known triggers."
        else:
            return "Could you share if you experience chest pain or coughing with phlegm?"

    # ------------------------- CONFIDENCE -------------------------
    def _calculate_confidence(
        self,
        audio_result: Optional[Dict],
        xray_result: Optional[Dict],
        full_context: Dict
    ) -> float:
        conf = 0.5
        tool_confidences = []
        if audio_result:
            tool_confidences.append(audio_result.get("confidence", 0.5))
        if xray_result:
            tool_confidences.append(xray_result.get("confidence", 0.5))
        if tool_confidences:
            conf = max(tool_confidences) * 0.6
        if full_context.get("patient_data"): conf += 0.15
        if len(full_context.get("disease_knowledge", [])) >= 3: conf += 0.15
        if full_context.get("patient_history"): conf += 0.10
        return min(conf, 1.0)

    # ------------------------- UTILITIES -------------------------
    def reset_memory(self):
        self.memory.clear()
        self.symptom_history.clear()
        self.conversation_turns.clear()
        logger.info("Conversation memory reset")

    def cleanup(self):
        self.retrieval_agent.cleanup()
        logger.info("LungScopeChatbot cleanup complete")
