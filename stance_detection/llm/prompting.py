from typing import Optional, List, Dict, Any, Type, TypeVar
from pydantic import BaseModel, Field
from enum import Enum

from .client import LLMClient, LLMConfig, LLMProvider, LLMError


class Stance(str, Enum):
    PRO = "pro"
    AGAINST = "against"
    NEUTRAL = "neutral"

class Sarcasm(str, Enum):
    SARCASM = 1
    NOT_SARCASM = 0

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class StanceDetectionResponse(BaseModel):
    """Structured response from the LLM for stance detection"""
    label: Stance = Field(
        description="The stance of the text towards the topic (pro, against, neutral)"
    )
    reasoning: str = Field(
        description="A brief explanation of why this label was chosen"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score between 0 and 1"
    )

class SarcasmDetectionResponse(BaseModel):
    label: Sarcasm = Field(
        description="The sarcasm of the text (1 or 0)"
    )
    reasoning: str = Field(
        description="A brief explanation of why this label was chosen"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score between 0 and 1"
    )

class SentimentDetectionResponse(BaseModel):
    label: Sentiment = Field(
        description="The sentiment of the text (positive, negative, neutral)"
    )
    reasoning: str = Field(
        description="A brief explanation of why this label was chosen"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score between 0 and 1"
    )

class AugmentationResponse(BaseModel):
    aug1: str = Field(
        description="The first paraphrased version of the text"
    )
    aug2: str = Field(
        description="The second paraphrased version of the text"
    )
    aug3: str = Field(
        description="The third paraphrased version of the text"
    )

SYSTEM_PROMPT_STANCE = """You are an expert linguistic analyst specializing in Arabic social media discourse and sentiment analysis. You are fluent in Modern Standard Arabic (MSA) and various regional dialects (e.g., Khaleeji, Levantine, Egyptian, Maghrebi)."""

SYSTEM_PROMPT = ""

LABELING_PROMPT = """Analyze the stance of the provided Arabic text toward the given topic. Categorize it into one of the following labels: pro, against, neutral.

### Contextual Guidelines
1. **Identify Sarcasm:** Arabic social media often uses irony or "black humor" to express opposition. Look for context clues that might indicate the user means the opposite of what they literally wrote.
2. **Handle Dialects:** Interpret regional idioms and slang correctly within the context of the topic.
3. **Neutrality:** Label as 'Neutral' if the post is purely informative, news-based without commentary, or contains equally balanced pros and cons.

### Input Data
- **Topic:** {topic}
- **Text:** {text}

### Output Format
Return ONLY a JSON object with the following keys:
- "label": (The chosen stance)
- "reasoning": (A brief, 1-sentence explanation in English of why this label was chosen)
- "confidence": (A score between 0 and 1)"""

SARCASM_PROMPT = """Analyze the sarcasm of the provided Arabic text. Categorize it into one of the following labels: sarcastic (label: 1), not sarcastic (label: 0).

### Input Data
- **Topic:** {topic}
- **Text:** {text}

### Output Format
Return ONLY a JSON object with the following keys:
- "label": (The chosen sarcasm label: 1 or 0)
- "reasoning": (A brief, 1-sentence explanation in English of why this label was chosen)
- "confidence": (A score between 0 and 1)
"""

SENTIMENT_PROMPT = """Analyze the sentiment of the provided Arabic text. Categorize it into one of the following labels: positive, negative, neutral.

### Input Data
- **Topic:** {topic}
- **Text:** {text}

### Output Format
Return ONLY a JSON object with the following keys:
- "label": (The chosen sentiment label: positive, negative, neutral)
- "reasoning": (A brief, 1-sentence explanation in English of why this label was chosen)
- "confidence": (A score between 0 and 1)
"""

AUG_PROMPT = """You are an expert data linguist specializing in Modern Standard Arabic (MSA) and Arabic dialects. You are augmenting a dataset for Stance Detection.

### Input Data:
- "text": The original social media post.
- "topic": The specific topic being discussed.
- "stance": The label (pro / against / neutral).

### Task:
Generate 3 paraphrased versions of the text that strictly preserve the stance and the topic.

### Strict Constraints:
1. **Topic Consistency:** The output MUST be about the provided "topic". Do not switch topics.
2. **Stance Preservation:** If the stance is for example "against", the paraphrased text's stance must remain "against".
3. **Dialect Integrity:**
   - If the input is dialectal, the paraphrase must be dialectal.
   - If the input is MSA, the paraphrase must be MSA.
   - If the input is mixed, the paraphrase must remain mixed.

### Output Format (JSON):
{{"aug1": "...", "aug2": "...", "aug3": "..."}}

### Data:
- **Topic:** {topic}
- **Text:** {text}
- **Stance:** {stance}
"""

T = TypeVar("T", bound=BaseModel)

TASK_CONFIG: Dict[str, Dict[str, Any]] = {
    "stance": {
        "prompt": LABELING_PROMPT,
        "response_model": StanceDetectionResponse,
    },
    "sarcasm": {
        "prompt": SARCASM_PROMPT,
        "response_model": SarcasmDetectionResponse,
    },
    "sentiment": {
        "prompt": SENTIMENT_PROMPT,
        "response_model": SentimentDetectionResponse,
    },
    "augmentation": {
        "prompt": AUG_PROMPT,
        "response_model": AugmentationResponse,
    },
}


def detect(
    topic: str,
    text: str,
    prompt_template: str,
    response_model: Type[T],
    client: Optional[LLMClient] = None,
    temperature: Optional[float] = None,
    **kwargs: str,
) -> T:
    llm = client or get_client()
    user_prompt = prompt_template.format(topic=topic, text=text, **kwargs)
    return llm.generate(
        response_format=response_model,
        user_prompt=user_prompt,
        system_prompt=SYSTEM_PROMPT,
        temperature=temperature,
    )


_client: Optional[LLMClient] = None


def get_client(config: Optional[LLMConfig] = None) -> LLMClient:
    global _client
    if _client is None:
        if config is not None:
            _client = LLMClient(config)
        else:
            _client = LLMClient.from_env()
    return _client


def set_client(client: LLMClient) -> None:
    global _client
    _client = client


def detect_stance(
    topic: str,
    text: str,
    client: Optional[LLMClient] = None,
    temperature: Optional[float] = None,
) -> StanceDetectionResponse:
    return detect(
        topic=topic,
        text=text,
        prompt_template=LABELING_PROMPT,
        response_model=StanceDetectionResponse,
        client=client,
        temperature=temperature,
    )

