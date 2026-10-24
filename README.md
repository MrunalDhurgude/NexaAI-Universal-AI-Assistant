# NexaAI – Smart NLP & General-Purpose AI Chatbot

**Course / Academic Project:** Artificial Intelligence and Data Science (AIDS) Engineering — 5th Semester  
**Project Title:** *Design and Development of an LLM-Based Chatbot Using Prompt Engineering Techniques*  
**Application Platform:** NexaAI Smart NLP & General AI Assistant  
**Frontend Framework:** Streamlit (Custom Modern Dashboard UI)  
**LLM Engine:** Google Gemini API (via official `google-genai` Python SDK)  

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Preserved Dashboard UI Design](#preserved-dashboard-ui-design)
3. [General-Purpose AI Capabilities](#general-purpose-ai-capabilities)
4. [Python Coding Assistant & Debugging](#python-coding-assistant--debugging)
5. [Prompt Engineering Architecture](#prompt-engineering-architecture)
6. [Multi-Turn Conversation Memory](#multi-turn-conversation-memory)
7. [System Instructions & Learning Level Adaptation](#system-instructions--learning-level-adaptation)
8. [Setup & Execution Guide](#setup--execution-guide)
9. [Demonstration & Testing Instructions](#demonstration--testing-instructions)
10. [Error Handling & API Security](#error-handling--api-security)
11. [Viva Voce Preparation](#viva-voce-preparation)

---

## 1. Project Overview

**NexaAI** is a general-purpose AI chatbot powered by the **Google Gemini API** and built in **Python** using **Streamlit**. Upgraded from a specialized NLP micro-project into a full-scale AI assistant, NexaAI answers questions ranging from basic conceptual inquiries to advanced technical problems across:
- **Python Programming:** Code generation, line-by-line explanations, bug diagnosis, and time/space complexity analysis.
- **Computer Science & Engineering:** Data structures, algorithms, operating systems, and distributed systems.
- **Mathematics & Logical Reasoning:** Step-by-step mathematical problem solving and formal proofs.
- **Prompt Engineering:** Dynamic Zero-Shot, Role-Based, Few-Shot, and Structured-Output prompting.
- **NLP Specialized Tasks:** General Q&A, Text Summarization, and 3-Class Sentiment Analysis.

---

## 2. Preserved Dashboard UI Design

The existing **NexaAI Dashboard** UI is preserved with complete fidelity:
- **Branding:** NexaAI robot badge with gradient brand title (*"NexaAI – Smart NLP Chatbot"*).
- **Sidebar:** Dark navy/indigo gradient with wave aesthetic and menu items:
  - 💬 **Chatbot** (Interactive conversation dashboard)
  - ⚖️ **Prompt Comparison** (Multi-technique matrix comparison)
  - 🗄️ **Dataset Testing** (Batch evaluation and accuracy computation)
  - 📊 **Evaluation** (Human rubric analytics, latency benchmarks, and academic report)
- **Top Status Indicator:** Real-time Gemini API status pill with pulsating indicator (`● Connected` in green, `● Key Missing` in amber, or `● Quota Limit`).
- **Hero Banner:** `✨ Your Intelligent AI Assistant` with subtext.
- **4 Feature Cards:** AI Chat (blue), Summarization (green), Sentiment Analysis (pink), and Prompt Engineering (purple).
- **Example Cards:** Clickable examples (*"Explain Artificial Intelligence"*, *"Summarize this text"*, *"Analyze sentiment"*).
- **Right Control Panel:** Task selection dropdown (`General Q&A`, `Python Programming & Debugging`, `Text Summarization`, `Sentiment Analysis`), Prompt Technique dropdown, Chat History list, and Clear Chat button.
- **Academic Badge:** Pinned footer for AIDS Engineering 5th Semester NLP Micro Project.

---

## 3. General-Purpose AI Capabilities

Users can ask any natural language question without being restricted to fixed categories:
- **General Knowledge:** Everyday facts, history, science, geography, and world events.
- **Academic & Engineering Concepts:** Machine learning, neural networks, computer architecture, databases, and network protocols.
- **Algorithms & Data Structures:** Step-by-step explanations of sorting, searching, trees, dynamic programming, and graphs.
- **Tutorials & Step-by-Step Guides:** Structured learning sequences with headings, bullet points, and practical analogies.
- **Follow-Up Questions:** Remembers previous turns so users can ask contextual questions without repeating context.

---

## 4. Python Coding Assistant & Debugging

When requested for programming assistance, NexaAI adheres to rigorous engineering standards:
1. **Approach Breakdown:** Summarizes the algorithmic strategy prior to writing code.
2. **Clean Python Code:** Generates readable, PEP 8-compliant code in syntax-highlighted Markdown code blocks with docstrings and type hints.
3. **Line-by-Line Explanation:** Explains non-obvious logic, helper functions, and edge cases.
4. **Complexity Analysis:** States Big-O Time Complexity and Space Complexity.
5. **Sample Input / Output:** Demonstrates functional test cases.
6. **Debugging Assistant:** When provided with error tracebacks or faulty code, identifies the root cause and provides the corrected solution.

---

## 5. Prompt Engineering Architecture

NexaAI implements four distinct prompt engineering techniques via [`prompts.py`](file:///c:/Users/mruna/OneDrive/Desktop/NLP_chatbot/prompts.py):

| Technique | Description | Impact on Response |
| :--- | :--- | :--- |
| **Zero-Shot (Basic)** | Direct task instruction without examples or persona framing. | Evaluates base parametric capability with minimum token overhead. |
| **Role-Based (Persona)** | Explicit persona conditioning (e.g., *Senior Python Architect*, *AI Professor*). | Calibrates technical depth, pedagogical clarity, and code quality. |
| **Few-Shot (Examples)** | Injects high-quality demonstration exemplars of coding and conceptual answers. | Primes in-context learning for consistent formatting, complexity analysis, and structure. |
| **Structured-Output (JSON)** | Enforces strict, machine-readable JSON schemas with typed keys. | Produces deterministic fields (`topic`, `summary`, `code_solution`, `time_complexity`) for API pipelines. |

---

## 6. Multi-Turn Conversation Memory

Implemented in [`llm_service.py`](file:///c:/Users/mruna/OneDrive/Desktop/NLP_chatbot/llm_service.py) and [`app.py`](file:///c:/Users/mruna/OneDrive/Desktop/NLP_chatbot/app.py):
- Maintains user and assistant message turns in Streamlit `st.session_state.chat_history`.
- Supplies a sliding window of recent conversation turns to the Google GenAI SDK (`google.genai`), enabling natural follow-up queries.
- Bounded to the last 8 messages to prevent exceeding model context windows.
- The **Clear Chat** button completely resets conversation memory.

---

## 7. System Instructions & Learning Level Adaptation

NexaAI operates under the following professional system instructions:

> *“You are NexaAI, a knowledgeable, helpful, and professional general-purpose AI assistant.*  
> *Answer user questions across programming, Python, computer science, mathematics, engineering, academics, and general knowledge.*  
> *Adapt your response to the user's selected learning level:*  
> *- Beginner: Use simple English, basic concepts, and easy examples.*  
> *- Intermediate: Explain concepts with practical examples and technical details.*  
> *- Advanced: Provide in-depth explanations, algorithms, technical reasoning, and relevant trade-offs.*  
> *For programming requests, provide complete and readable code when requested, explain the approach, use appropriate formatting, and include examples where helpful. When debugging, identify the likely cause of the error and provide a corrected solution.*  
> *For conceptual questions, provide accurate explanations using headings, bullet points, and step-by-step reasoning when appropriate.*  
> *Maintain relevant conversation context so users can ask follow-up questions without repeating all previous information.*  
> *Follow user instructions about the requested output format. If a question is ambiguous, ask for clarification. If you are uncertain, acknowledge the uncertainty instead of inventing facts.*  
> *Be concise for simple questions and detailed for complex questions. Remain polite, professional, and focused on solving the user's problem.”*

---

## 8. Setup & Execution Guide

### Step 1: Install Requirements
```powershell
pip install -r requirements.txt
```

### Step 2: Configure Gemini API Key
Obtain a free API key from [Google AI Studio](https://aistudio.google.com/).  
You can configure your key in two ways:
1. **Via `.env` file:** Create a `.env` file in the project root:
   ```env
   GEMINI_API_KEY=AIzaSy...your_actual_key_here
   GEMINI_MODEL=gemini-3-flash-preview
   ```
   *(Recommended active models: `gemini-3-flash-preview`, `gemini-3.1-flash-lite-preview`, `gemini-3.5-flash`)*
2. **Via Sidebar Settings:** Paste the key directly into the **⚙️ Gemini API & Settings** expander in the left sidebar at runtime.

### Step 3: Run the Application
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 9. Demonstration & Testing Instructions

### Test Case 1: General Q&A and Conceptual Explanation
- **Input:** *"Explain the difference between supervised and unsupervised learning with real-world examples."*
- **Observation:** NexaAI breaks down definitions, comparisons, real-world examples, and trade-offs.

### Test Case 2: Python Code Generation & Complexity
- **Input:** *"Write a Python function to find the longest substring without repeating characters."*
- **Observation:** NexaAI provides the sliding window approach, clean code with type hints, sample input/output, and $O(N)$ time / space complexity.

### Test Case 3: Multi-Turn Follow-Up Query
- **Follow-Up Input:** *"Can you explain lines 8-12 of that code in detail and optimize the space usage?"*
- **Observation:** NexaAI references the previous code without needing the user to paste it again.

### Test Case 4: Python Debugging Assistant
- **Input:**  
  ```python
  def calculate_average(nums):
      total = 0
      for i in range(len(nums) + 1):
          total += nums[i]
      return total / len(nums)
  ```
  *"Why is this giving an IndexError?"*
- **Observation:** NexaAI diagnoses the off-by-one error (`range(len(nums) + 1)`), explains index out of range, and provides the corrected Python code.

### Test Case 5: Prompt Comparison Matrix
- Navigate to **⚖️ Prompt Comparison** in the left sidebar.
- Enter any question or coding problem and click **Run Comparative Benchmark** to view outputs from all 4 techniques side-by-side.

---

## 10. Error Handling & API Security

- **Missing API Key:** When no key is entered, NexaAI displays a friendly instruction banner explaining how to configure the key, rather than outputting fake responses.
- **Quota Limit (HTTP 429):** Notifies the user when rate limits are reached, recommending a brief wait.
- **Security:** API keys are never displayed in chat bubbles, source code, or exported CSV files.

---

## 11. Viva Voce Preparation

1. **How does NexaAI differ from fixed-intent rule-based chatbots?**  
   *Answer:* Rule-based chatbots use hardcoded regex patterns or static intents. NexaAI leverages a foundation Large Language Model (Google Gemini) guided by prompt engineering, enabling open-domain comprehension, coding assistance, and multi-turn reasoning.

2. **How is conversation memory handled in Streamlit?**  
   *Answer:* Streamlit reruns scripts on interaction. NexaAI stores conversational turns in `st.session_state.chat_history` and injects recent turns into the Gemini API `contents` payload to preserve context across turns.

3. **What role does Few-Shot prompting play in programming requests?**  
   *Answer:* Few-shot exemplars demonstrate to the model how to structure code solutions (approach, code block, line breakdown, and complexity analysis), resulting in consistent, high-quality responses without model retraining.
