"""
diagnostic_tool_selector.py - Deterministic routing for AIRA diagnostic tools.

This module decides which diagnostic upload the chatbot should request based on
recent symptom text. The LLM still writes the response, but it must follow this
tool decision.
"""

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class DiagnosticToolDecision:
    action: str
    symptom_focus: str
    reason: str
    detail_questions: List[str]
    upload_instruction: str
    should_request_audio: bool
    should_request_xray: bool
    urgency: str = "routine"
    audio_score: int = 0
    xray_score: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)


class DeterministicDiagnosticToolSelector:
    """Keyword and phrase based diagnostic routing."""

    GREETING_TERMS: Tuple[str, ...] = (
        "hi",
        "hello",
        "hey",
        "hey there",
        "good morning",
        "good afternoon",
        "good evening",
        "how are you",
        "what's up",
        "whats up",
    )

    AUDIO_TERMS: Tuple[Tuple[str, int], ...] = (
        ("wheez", 4),
        ("crackle", 4),
        ("rattle", 3),
        ("whistling", 3),
        ("noisy breathing", 3),
        ("breath sound", 4),
        ("lung sound", 4),
        ("shortness of breath", 3),
        ("breathless", 3),
        ("difficulty breathing", 3),
        ("can't take deep breath", 3),
        ("chest tightness", 2),
        ("asthma", 3),
        ("copd", 3),
        ("bronchitis", 2),
        ("cough", 2),
        ("phlegm", 2),
        ("mucus", 2),
        ("sputum", 2),
        ("inhaler", 2),
        ("respiratory", 1),
    )

    XRAY_TERMS: Tuple[Tuple[str, int], ...] = (
        ("x-ray", 5),
        ("xray", 5),
        ("radiograph", 5),
        ("chest film", 5),
        ("scan image", 3),
        ("x ray", 5),
        ("pneumonia", 4),
        ("lung opacity", 5),
        ("opacity", 4),
        ("infiltrate", 4),
        ("consolidation", 4),
        ("viral pneumonia", 5),
        ("pleural", 3),
        ("radiology", 4),
        ("report says", 3),
        ("shadow", 3),
        ("patch", 2),
        ("fever", 2),
        ("high fever", 3),
        ("chills", 2),
        ("chest pain", 2),
        ("pain when breathing", 3),
        ("sharp pain", 2),
        ("fast breathing", 2),
        ("rapid breathing", 2),
        ("shallow breathing", 2),
        ("night sweats", 2),
        ("loss of appetite", 1),
        ("very tired", 1),
        ("fatigue", 1),
        ("green phlegm", 2),
        ("yellow phlegm", 2),
        ("brown phlegm", 2),
        ("persistent cough", 2),
    )

    URGENT_TERMS: Tuple[str, ...] = (
        "can't breathe",
        "cannot breathe",
        "severe shortness of breath",
        "severe breathlessness",
        "blue lips",
        "bluish lips",
        "confusion",
        "fainting",
        "passed out",
        "coughing blood",
        "blood in sputum",
        "hemoptysis",
        "oxygen below 92",
        "spo2 below 92",
        "oxygen saturation below 92",
        "crushing chest pain",
        "severe chest pain",
    )

    MEDICAL_TERMS: Tuple[str, ...] = tuple(
        sorted(
            {
                term
                for term, _ in AUDIO_TERMS + XRAY_TERMS
            }
            | {
                "sick",
                "ill",
                "unwell",
                "pain",
                "symptom",
                "symptoms",
                "medicine",
                "medication",
                "doctor",
                "hospital",
                "clinic",
                "diagnosis",
                "infection",
                "cold",
                "flu",
                "breath",
                "breathe",
                "breathing",
                "lung",
                "lungs",
                "throat",
                "tired",
                "weak",
                "dizzy",
                "headache",
            }
        )
    )

    def select(
        self,
        recent_messages: Iterable[str],
        has_audio_result: bool = False,
        has_xray_result: bool = False,
    ) -> DiagnosticToolDecision:
        raw_text = " ".join(message or "" for message in recent_messages).lower()
        text = self._normalize_symptom_text(raw_text)

        if not has_audio_result and not has_xray_result and self._is_general_conversation(text):
            return self._decision(
                action="conversation",
                focus="general_conversation",
                reason="No medical symptoms or diagnostic intent detected.",
                questions=[],
                upload="Do not ask for diagnostic uploads. Continue normal conversation and invite the patient to share symptoms if they need help.",
                audio_score=0,
                xray_score=0,
            )

        if has_audio_result and has_xray_result:
            return self._decision(
                action="none",
                focus="completed",
                reason="Both diagnostic tool outputs are already available.",
                questions=[],
                upload="No additional upload is needed right now.",
                audio_score=0,
                xray_score=0,
            )

        if any(term in raw_text for term in self.URGENT_TERMS) or self._has_low_oxygen(raw_text):
            return self._decision(
                action="urgent_care",
                focus="red_flags",
                reason="The symptom text contains urgent warning signs.",
                questions=[
                    "Are you having severe breathlessness, chest pain, confusion, blue lips, or oxygen saturation below 92 percent?",
                ],
                upload="Do not wait for an upload; seek urgent medical care now.",
                urgency="urgent",
            )

        audio_score = self._score(text, self.AUDIO_TERMS)
        xray_score = self._score(text, self.XRAY_TERMS)

        lower_infection_bonus = self._lower_respiratory_infection_bonus(text)
        xray_score += lower_infection_bonus
        if self._has_airway_pattern(text):
            audio_score += 3

        if has_audio_result:
            audio_score = 0
        if has_xray_result:
            xray_score = 0

        if audio_score >= 4 and xray_score >= 4:
            return self._decision(
                action="both",
                focus="mixed_airway_and_imaging",
                reason="Symptoms include both airway/lung sound clues and lower-respiratory infection or imaging-relevant clues.",
                questions=[
                    "How long have the cough, fever, chest pain, or breathing symptoms been present, and are they worsening?",
                    "Do you hear wheezing, crackling, or noisy breathing?",
                    "Do you have fever, chills, pain when breathing, or a chest X-ray image from a recent visit?",
                ],
                upload="Ask for a lung sound recording and ask for a chest X-ray image if available. If only one can be uploaded now, either is acceptable.",
                audio_score=audio_score,
                xray_score=xray_score,
            )

        if audio_score >= 4 and audio_score >= xray_score + 1:
            return self._decision(
                action="audio",
                focus="airway_or_lung_sounds",
                reason="Symptoms are most consistent with airway/lung sound assessment.",
                questions=[
                    "Do you hear wheezing, crackling, or noisy breathing?",
                    "How severe is the shortness of breath at rest and while walking?",
                    "Is there cough, phlegm, chest tightness, or an asthma/COPD history?",
                ],
                upload="Ask for a 10-15 second lung sound or breathing audio recording.",
                audio_score=audio_score,
                xray_score=xray_score,
            )

        if xray_score >= 4 and xray_score >= audio_score + 1:
            return self._decision(
                action="xray",
                focus="lower_respiratory_infection_or_imaging",
                reason="The symptom pattern suggests a lower-respiratory infection or another imaging-relevant concern.",
                questions=[
                    "Do you have fever, chills, chest pain, or pain when taking a deep breath?",
                    "How many days have the cough or breathing symptoms lasted, and are they worsening?",
                    "Do you have a chest X-ray image from a recent visit that you can upload?",
                ],
                upload="Ask for a chest X-ray image if available; if not available, advise clinical review to decide whether chest imaging is needed.",
                audio_score=audio_score,
                xray_score=xray_score,
            )

        return self._decision(
            action="flexible",
            focus="unclear_or_general_respiratory",
            reason="Symptoms do not strongly match only audio or only X-ray routing.",
            questions=[
                "Is your main concern breathing sounds or trouble breathing?",
                "Do you have fever, chest pain, cough, phlegm, wheezing, or symptoms that are worsening?",
                "Do you have a chest X-ray image from a recent visit, or can you record lung sounds?",
            ],
            upload="Ask for either a lung sound recording or a chest X-ray image if available, whichever the patient can provide.",
            audio_score=audio_score,
            xray_score=xray_score,
        )

    def _score(self, text: str, weighted_terms: Tuple[Tuple[str, int], ...]) -> int:
        return sum(weight for term, weight in weighted_terms if term in text)

    def _is_general_conversation(self, text: str) -> bool:
        cleaned = " ".join(text.replace(".", " ").replace(",", " ").replace("!", " ").replace("?", " ").split())
        if not cleaned:
            return True

        has_medical_term = any(term in cleaned for term in self.MEDICAL_TERMS)
        if not has_medical_term:
            return True

        return False

    def _normalize_symptom_text(self, text: str) -> str:
        negated_phrases = (
            "no fever",
            "no high fever",
            "no temperature",
            "no high temperature",
            "without fever",
            "without a fever",
            "don't have fever",
            "do not have fever",
            "doesn't have fever",
            "not feverish",
            "no chest pain",
            "without chest pain",
            "no pain when breathing",
            "no phlegm",
            "no mucus",
            "no sputum",
            "no wheezing",
            "without wheezing",
        )
        normalized = text
        for phrase in negated_phrases:
            normalized = normalized.replace(phrase, " ")
        return normalized

    def _has_low_oxygen(self, text: str) -> bool:
        oxygen_terms = ("spo2", "oxygen", "oxygen saturation", "o2")
        if not any(term in text for term in oxygen_terms):
            return False
        for marker in ("91", "90", "89", "88", "87", "86", "85"):
            if marker in text:
                return True
        return False

    def _lower_respiratory_infection_bonus(self, text: str) -> int:
        has_fever = any(term in text for term in ("fever", "high temperature", "temperature", "chills", "night sweats"))
        has_cough = any(term in text for term in ("cough", "phlegm", "mucus", "sputum"))
        has_chest_symptom = any(term in text for term in ("chest pain", "pain when breathing", "sharp pain", "tight chest"))
        has_breathing_symptom = any(term in text for term in ("shortness of breath", "breathless", "fast breathing", "rapid breathing", "shallow breathing"))
        has_duration_or_worse = any(term in text for term in ("worse", "worsening", "persistent", "days", "week", "not improving"))
        has_colored_sputum = any(term in text for term in ("green phlegm", "yellow phlegm", "brown phlegm", "green mucus", "yellow mucus"))

        bonus = 0
        if has_fever and has_cough and (has_chest_symptom or has_breathing_symptom):
            bonus += 6
        elif has_fever and has_cough:
            bonus += 4

        if has_cough and has_chest_symptom and has_duration_or_worse:
            bonus += 4
        elif has_cough and has_duration_or_worse:
            bonus += 2

        if has_colored_sputum and has_fever:
            bonus += 3

        return bonus

    def _has_airway_pattern(self, text: str) -> bool:
        has_breathing = any(term in text for term in ("wheez", "breathless", "shortness of breath", "difficulty breathing", "chest tightness"))
        has_airway_history = any(term in text for term in ("asthma", "copd", "bronchitis", "inhaler"))
        return has_breathing or has_airway_history

    def _decision(
        self,
        action: str,
        focus: str,
        reason: str,
        questions: List[str],
        upload: str,
        urgency: str = "routine",
        audio_score: int = 0,
        xray_score: int = 0,
    ) -> DiagnosticToolDecision:
        return DiagnosticToolDecision(
            action=action,
            symptom_focus=focus,
            reason=reason,
            detail_questions=questions,
            upload_instruction=upload,
            should_request_audio=action in {"audio", "both", "flexible"},
            should_request_xray=action in {"xray", "both", "flexible"},
            urgency=urgency,
            audio_score=audio_score,
            xray_score=xray_score,
        )
