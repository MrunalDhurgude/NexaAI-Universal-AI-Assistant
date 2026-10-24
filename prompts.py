"""
prompts.py - Advanced Prompt Engineering Module for NexaAI.

Implements the official professional system instructions and dynamic prompt construction
for general-purpose AI assistance, including:
- General knowledge & academic engineering concepts
- Python programming, code generation, line-by-line explanations, and debugging
- Algorithmic explanations and time/space complexity
- Multi-technique prompt engineering:
  1. Zero-Shot Prompting (Basic)
  2. Role-Based Prompting (Persona & Level Adaptation)
  3. Few-Shot Prompting (Demonstration Exemplars)
  4. Structured-Output Prompting (Strict JSON Schemas)
"""

from typing import Dict, Tuple, List, Optional
import json
import re

# -----------------------------------------------------------------------------
# 1. CONSTANTS & DEFINITIONS
# -----------------------------------------------------------------------------

TECHNIQUE_ZERO_SHOT = "Zero-Shot Prompting"
TECHNIQUE_ROLE_BASED = "Role-Based Prompting"
TECHNIQUE_FEW_SHOT = "Few-Shot Prompting"
TECHNIQUE_STRUCTURED_OUTPUT = "Structured-Output Prompting"

ALL_TECHNIQUES = [
    TECHNIQUE_ZERO_SHOT,
    TECHNIQUE_ROLE_BASED,
    TECHNIQUE_FEW_SHOT,
    TECHNIQUE_STRUCTURED_OUTPUT,
]

TASK_QA = "qa"
TASK_CODE = "code"
TASK_SUMMARIZATION = "summarization"
TASK_SENTIMENT = "sentiment"

TASK_DISPLAY_NAMES = {
    TASK_QA: "General Q&A",
    TASK_CODE: "Python Code & Debugging",
    TASK_SUMMARIZATION: "Text Summarization",
    TASK_SENTIMENT: "Sentiment Analysis",
}

# -----------------------------------------------------------------------------
# 2. OFFICIAL GENERAL-PURPOSE SYSTEM INSTRUCTION
# -----------------------------------------------------------------------------

CORE_SYSTEM_INSTRUCTION = """You are NexaAI, a helpful, knowledgeable, and professional general-purpose AI assistant. Answer questions across programming, Python, computer science, mathematics, engineering, academics, science, and general knowledge.
Adapt your explanations to the user's selected learning level.
For programming requests, provide complete and readable code when requested, use correct syntax and meaningful variable names, explain the approach, and include sample input and output where useful. When debugging, identify the likely cause, provide a corrected version, and explain the changes. Never claim that code was executed or tested unless it actually was.
For conceptual questions, explain topics clearly using practical examples and step-by-step reasoning where appropriate. Use headings, lists, and code blocks to improve readability. Ask for clarification when necessary, acknowledge uncertainty, and avoid inventing facts.
Maintain relevant context from the current conversation and answer follow-up questions consistently. Do not execute generated code automatically."""


# Task-specific system instruction modifiers
TASK_SYSTEM_MODIFIERS: Dict[str, str] = {
    TASK_QA: (
        "Focus on providing comprehensive, accurate, and clearly articulated answers across science, mathematics, "
        "engineering, computer science, and academics. Use structured headings and step-by-step reasoning."
    ),
    TASK_CODE: (
        "Focus on Python programming, algorithms, code generation, line-by-line explanation, and debugging. "
        "Provide complete, readable, and PEP 8 compliant code with meaningful variable names and type hints. "
        "When debugging, pinpoint the exact cause of bugs, provide corrected code, and explain changes clearly."
    ),
    TASK_SUMMARIZATION: (
        "Focus on condensing the user's input text into its essential points while preserving "
        "key facts, numbers, and core takeaways. Use bullet points or executive summaries where appropriate."
    ),
    TASK_SENTIMENT: (
        "Focus on classifying the sentiment of the text strictly into Positive, Negative, or Neutral. "
        "Provide a confidence assessment and brief textual evidence or reasoning."
    )
}

# -----------------------------------------------------------------------------
# 3. FEW-SHOT DEMONSTRATION EXEMPLARS (General, Python & Conceptual)
# -----------------------------------------------------------------------------

FEW_SHOT_GENERAL_EXEMPLARS: List[Dict[str, str]] = [
    {
        "input": "Write a Python function to find two numbers in a list that add up to a target sum.",
        "output": (
            "### Approach\n"
            "We can solve the Two-Sum problem efficiently using a hash map (dictionary) in Python. "
            "As we iterate through the array, we check if the complement (`target - num`) has already been seen.\n\n"
            "### Python Solution\n"
            "```python\n"
            "def two_sum(nums: list[int], target: int) -> list[int]:\n"
            "    \"\"\"\n"
            "    Returns indices of the two numbers such that they add up to target.\n"
            "    \"\"\"\n"
            "    seen: dict[int, int] = {}  # maps value to index\n"
            "    for index, num in enumerate(nums):\n"
            "        complement = target - num\n"
            "        if complement in seen:\n"
            "            return [seen[complement], index]\n"
            "        seen[num] = index\n"
            "    return []\n"
            "\n"
            "# Example Usage:\n"
            "numbers = [2, 7, 11, 15]\n"
            "target_value = 9\n"
            "result = two_sum(numbers, target_value)\n"
            "print(\"Indices:\", result)  # Output: [0, 1]\n"
            "```\n\n"
            "### Complexity Analysis\n"
            "- **Time Complexity:** $O(N)$ because dictionary lookups and insertions take $O(1)$ average time.\n"
            "- **Space Complexity:** $O(N)$ to store up to $N$ elements in the hash map."
        )
    },
    {
        "input": "Explain the difference between synchronous and asynchronous execution in computer science.",
        "output": (
            "### 1. Synchronous Execution\n"
            "In synchronous operations, tasks are executed sequentially. The program must wait for a current task "
            "(such as a disk read or network query) to finish before moving to the next line of code.\n"
            "- **Analogy:** Waiting in a single queue at a grocery store checkout.\n"
            "- **Pros:** Simple, deterministic, and easy to trace.\n"
            "- **Cons:** Can block execution and waste CPU cycles during I/O operations.\n\n"
            "### 2. Asynchronous Execution\n"
            "In asynchronous operations, a task can be initiated in the background, allowing the main program "
            "to continue processing other events. When the background task finishes, a callback or future is resolved.\n"
            "- **Analogy:** Ordering at a coffee shop, receiving a buzzer, and sitting down until the coffee is ready.\n"
            "- **Pros:** High concurrency, non-blocking I/O, ideal for modern web servers and UI responsiveness.\n"
            "- **Cons:** Increased architectural complexity and potential race conditions."
        )
    }
]

FEW_SHOT_SENTIMENT_EXEMPLARS: List[Dict[str, str]] = [
    {
        "input": "The build quality and audio clarity exceeded all my expectations. The battery lasted 30 hours straight on my flight!",
        "output": "Sentiment: Positive\nExplanation: Expresses high satisfaction with device battery performance and audio clarity."
    },
    {
        "input": "The latest software update completely broke the camera app and drains my battery in two hours.",
        "output": "Sentiment: Negative\nExplanation: Reports severe operational failure, crashes, and rapid battery discharge."
    },
    {
        "input": "The university library opens daily at 8:00 AM and closes at 10:00 PM on weekdays.",
        "output": "Sentiment: Neutral\nExplanation: States an objective, factual operating schedule with no emotional tone."
    }
]

FEW_SHOT_SUMMARIZATION_EXEMPLARS: List[Dict[str, str]] = [
    {
        "input": (
            "Artificial intelligence is revolutionizing healthcare by assisting radiologists in detecting early-stage "
            "tumors from MRI scans with superhuman speed. Hospitals adopting automated diagnostic pipelines report a 35% "
            "reduction in diagnostic backlog. However, regulatory authorities insist that human physicians must retain "
            "ultimate decision-making authority to avoid liability and algorithmic misdiagnosis."
        ),
        "output": (
            "**Executive Summary:**\n"
            "- **Impact:** AI assists radiologists in detecting early-stage tumors and cuts hospital diagnostic backlogs by 35%.\n"
            "- **Constraint:** Regulatory authorities require human doctors to retain final diagnostic authority to prevent misdiagnosis."
        )
    }
]

# -----------------------------------------------------------------------------
# 4. STRUCTURED OUTPUT SCHEMAS
# -----------------------------------------------------------------------------

STRUCTURED_JSON_SCHEMA = """{
  "topic": "Brief topic or problem title",
  "summary": "Direct answer or executive summary (1-3 sentences)",
  "detailed_explanation": "In-depth explanation with step-by-step reasoning or conceptual breakdown",
  "code_solution": {
    "language": "python",
    "code": "# Python code if applicable, otherwise empty string",
    "explanation": "Line-by-line or approach explanation",
    "time_complexity": "O(...) if algorithmic, else N/A",
    "space_complexity": "O(...) if algorithmic, else N/A"
  },
  "key_takeaways": [
    "Core takeaway 1",
    "Core takeaway 2",
    "Core takeaway 3"
  ],
  "learning_level": "Beginner | Intermediate | Advanced"
}"""

STRUCTURED_SENTIMENT_SCHEMA = """{
  "task": "Sentiment Analysis",
  "sentiment": "Positive | Negative | Neutral",
  "confidence_score": 0.0 to 1.0,
  "reasoning": "Linguistic and semantic justification",
  "key_emotional_words": ["word1", "word2"]
}"""

STRUCTURED_SUMMARIZATION_SCHEMA = """{
  "task": "Text Summarization",
  "concise_summary": "1 to 2 sentence overall summary",
  "key_bullet_points": [
    "Key finding 1",
    "Key finding 2",
    "Key finding 3"
  ],
  "original_word_count_estimate": 100,
  "summary_word_count_estimate": 30
}"""

# -----------------------------------------------------------------------------
# 5. DYNAMIC PROMPT BUILDER
# -----------------------------------------------------------------------------

def get_technique_description(technique: str) -> str:
    """Returns a clear educational summary of the prompt engineering technique."""
    descriptions = {
        TECHNIQUE_ZERO_SHOT: (
            "Direct task instruction without demonstration examples or persona framing. "
            "Evaluates raw parametric knowledge and base instruction-following capabilities."
        ),
        TECHNIQUE_ROLE_BASED: (
            "Explicit persona framing (e.g. Senior Python Engineer, AI Professor). "
            "Guides depth, pedagogical tone, and real-world engineering standards."
        ),
        TECHNIQUE_FEW_SHOT: (
            "Supplies high-quality demonstration exemplars showing the exact reasoning pattern, "
            "code formatting, and technical depth desired (In-Context Learning)."
        ),
        TECHNIQUE_STRUCTURED_OUTPUT: (
            "Enforces a strict, machine-readable JSON schema with explicit keys for explanation, "
            "code solution, complexity, and key takeaways."
        ),
    }
    return descriptions.get(technique, "Standard prompt execution.")


def _detect_code_intent(text: str) -> bool:
    """Heuristically detects whether a user prompt is asking for code or debugging."""
    code_keywords = [
        "python", "code", "function", "script", "program", "algorithm",
        "debug", "error", "traceback", "class", "def ", "import ", "regex",
        "complexity", "two sum", "bubble sort", "binary search", "recursion"
    ]
    lower = text.lower()
    return any(kw in lower for kw in code_keywords)


def build_prompt(
    task_type: str,
    technique: str,
    user_input: str,
    learning_level: str = "Intermediate",
    custom_role: Optional[str] = None
) -> Tuple[str, str]:
    """
    Constructs the system instruction and user prompt dynamically based on the
    task type, prompt engineering technique, and user learning level.

    Args:
        task_type: 'qa' (General Q&A), 'summarization', or 'sentiment'.
        technique: One of the 4 prompt engineering techniques.
        user_input: The user's question, code request, or text.
        learning_level: 'Beginner', 'Intermediate', or 'Advanced'.
        custom_role: Optional custom role override.

    Returns:
        Tuple of (system_instruction, user_prompt)
    """
    task_type = task_type.lower()
    clean_input = user_input.strip()
    is_code = _detect_code_intent(clean_input)

    # Base System Instruction with level guidance
    system_instruction = CORE_SYSTEM_INSTRUCTION
    task_modifier = TASK_SYSTEM_MODIFIERS.get(task_type, "")
    if task_modifier:
        system_instruction = f"{system_instruction}\n\nTask Focus: {task_modifier}"

    system_instruction += f"\nTarget User Level: {learning_level}."

    # 1. ZERO-SHOT PROMPTING
    if technique == TECHNIQUE_ZERO_SHOT:
        if task_type == TASK_SENTIMENT:
            user_prompt = (
                f"Analyze the sentiment of the following text.\n"
                f"Classify it as Positive, Negative, or Neutral.\n\n"
                f"Text: \"{clean_input}\"\n\n"
                f"Sentiment:"
            )
        elif task_type == TASK_SUMMARIZATION:
            user_prompt = (
                f"Summarize the following text concisely, highlighting the main takeaways:\n\n"
                f"{clean_input}\n\n"
                f"Summary:"
            )
        else:  # General Q&A / Python / Math / CS
            user_prompt = clean_input

    # 2. ROLE-BASED PROMPTING
    elif technique == TECHNIQUE_ROLE_BASED:
        if custom_role:
            role_persona = custom_role
        elif is_code:
            role_persona = (
                "You are a Senior Principal Python Architect & Staff Software Engineer. "
                "Provide clean, production-grade Python code adhering to PEP 8 standards, "
                "include informative docstrings and type hints, explain your approach step-by-step, "
                "and clearly state the time and space complexity."
            )
        elif task_type == TASK_SENTIMENT:
            role_persona = (
                "You are a Senior NLP Data Scientist analyzing customer sentiment and textual polarity. "
                "Identify nuance, sarcasm, and emotional cues with objective precision."
            )
        elif task_type == TASK_SUMMARIZATION:
            role_persona = (
                "You are an Executive AI Research Editor. Provide an executive summary highlighting "
                "core findings, key metrics, and practical implications in crisp bullet points."
            )
        else:
            role_persona = (
                "You are an acclaimed Professor of Artificial Intelligence and Computer Science Engineering. "
                "Break down complex topics with clear technical explanations, intuitive analogies, "
                "and practical engineering examples."
            )

        system_instruction = f"{system_instruction}\n\nAssigned Persona: {role_persona}"

        if task_type == TASK_SENTIMENT:
            user_prompt = (
                f"[Role Instruction: Act as an expert NLP Data Scientist.]\n"
                f"Analyze the sentiment polarity of the following text with linguistic justification.\n\n"
                f"Input: \"{clean_input}\"\n\n"
                f"Classification & Justification:"
            )
        elif task_type == TASK_SUMMARIZATION:
            user_prompt = (
                f"[Role Instruction: Act as an Executive AI Research Editor.]\n"
                f"Provide a structured executive summary of the following text:\n\n"
                f"{clean_input}\n\n"
                f"Executive Summary:"
            )
        else:
            role_prefix = "[Role Instruction: Provide an authoritative, structured, and pedagogical explanation.]\n"
            user_prompt = f"{role_prefix}{clean_input}"

    # 3. FEW-SHOT PROMPTING
    elif technique == TECHNIQUE_FEW_SHOT:
        if task_type == TASK_SENTIMENT:
            exemplars = FEW_SHOT_SENTIMENT_EXEMPLARS
        elif task_type == TASK_SUMMARIZATION:
            exemplars = FEW_SHOT_SUMMARIZATION_EXEMPLARS
        else:
            exemplars = FEW_SHOT_GENERAL_EXEMPLARS

        exemplar_blocks = []
        for i, ex in enumerate(exemplars, 1):
            exemplar_blocks.append(f"### Example {i}:\nInput: {ex['input']}\n\nOutput:\n{ex['output']}")

        exemplars_text = "\n\n".join(exemplar_blocks)

        user_prompt = (
            f"Here are reference demonstration examples of high-quality, structured responses:\n\n"
            f"{exemplars_text}\n\n"
            f"### Now respond to this input following the same high standard of depth, formatting, and clarity:\n"
            f"Input: {clean_input}\n\n"
            f"Output:"
        )

    # 4. STRUCTURED-OUTPUT PROMPTING
    elif technique == TECHNIQUE_STRUCTURED_OUTPUT:
        if task_type == TASK_SENTIMENT:
            schema = STRUCTURED_SENTIMENT_SCHEMA
        elif task_type == TASK_SUMMARIZATION:
            schema = STRUCTURED_SUMMARIZATION_SCHEMA
        else:
            schema = STRUCTURED_JSON_SCHEMA

        system_instruction = (
            f"{system_instruction}\n\n"
            f"CRITICAL FORMAT REQUIREMENT: You MUST respond strictly in valid JSON format conforming "
            f"to the provided JSON schema. Do not enclose the output in conversational chatter outside the JSON."
        )

        user_prompt = (
            f"Analyze and respond to the following request strictly using this JSON schema:\n\n"
            f"JSON Schema:\n{schema}\n\n"
            f"User Input:\n{clean_input}\n\n"
            f"JSON Response:"
        )

    else:
        user_prompt = clean_input

    return system_instruction, user_prompt
