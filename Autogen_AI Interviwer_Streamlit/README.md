# 🎯 AI Interviewer - Streamlit MVP

A complete AI-powered interview system using AutoGen multi-agent orchestration. Practice technical interviews with real-time feedback, objective scoring, and comprehensive performance analysis.

---

## 📋 Table of Contents
    
1. [Overview](#overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Configuration](#configuration)
7. [Running the Application](#running-the-application)
8. [How to Use](#how-to-use)
9. [Technical Workflow](#technical-workflow)
10. [Agent System](#agent-system)
11. [Troubleshooting](#troubleshooting)
12. [Future Enhancements](#future-enhancements)

---

## 🎨 Overview

The AI Interviewer is a Streamlit-based MVP that simulates realistic technical interviews using multiple AI agents powered by AutoGen and OpenAI's GPT-4. The system provides:

- **Structured interviews** with progressive difficulty
- **Real-time coaching feedback** on your answers
- **Objective scoring** across multiple dimensions
- **Comprehensive performance reports**

### Why This Matters

- **Interview Preparation**: Practice in a safe, judgment-free environment
- **Immediate Feedback**: Get constructive criticism after every answer
- **Skill Assessment**: Track your performance across technical and soft skills
- **Flexible Practice**: Configure role, domain, and question count

---

## ✨ Features

### Core Capabilities

✅ **Multi-Agent AI System**
- AI Interviewer Agent (asks questions, controls flow)
- Coach Agent (provides feedback)
- Score Agent (objective evaluation)
- User Proxy Agent (represents you)

✅ **Smart Interview Flow**
- Progressive difficulty (easy → medium → hard)
- Context-aware questions based on role and domain
- Natural conversation flow

✅ **Comprehensive Feedback**
- Strengths identification
- Gap analysis
- Actionable suggestions

✅ **Objective Scoring**
- Technical Accuracy (0-10)
- Communication Clarity (0-10)
- Depth of Knowledge (0-10)
- Answer Relevance (0-10)
- Overall Score (average)

✅ **Rich UI/UX**
- Real-time interview interface
- Progress tracking
- Conversation history
- Performance analytics
- Downloadable reports

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│         Streamlit Frontend              │
│  ┌──────────────────────────────────┐  │
│  │    UI Components                 │  │
│  │  - Config Screen                 │  │
│  │  - Interview Interface           │  │
│  │  - Feedback Panel                │  │
│  │  - Summary Dashboard             │  │
│  └















{
    "question_id": 1,
    "technical_accuracy": 8,
    "communication": 7,
    "depth": 6,
    "relevance": 9,
    "total_score": 7.5,
    "justification": "..."
  }
```

### Communication Flow Diagram
```
User Action → Streamlit UI
                ↓
         Update session_state
                ↓
    Trigger AutoGen GroupChat
                ↓
    ┌───────────────────────────┐
    │   Interviewer Agent       │
    │   Posts next question     │
    └───────────────────────────┘
                ↓
         Display in UI
                ↓
         User types answer
                ↓
    ┌───────────────────────────┐
    │   User Proxy Agent        │
    │   Submits answer          │
    └───────────────────────────┘
                ↓
         Parallel Processing
    ┌─────────────┬─────────────┐
    │ Coach Agent │ Score Agent │
    │ Analyzes    │ Evaluates   │
    │ Provides    │ Scores in   │
    │ Feedback    │ JSON        │
    └─────────────┴─────────────┘
                ↓
    Store feedback & scores in session_state
                ↓
         Display in UI panels
                ↓
    Interviewer posts next question
    (or ends interview)


    1. High-Level Streamlit MVP Architecture
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit Frontend                       │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  UI Layer (st.components)                             │  │
│  │  - Interview Config Screen                            │  │
│  │  - Chat Interface                                     │  │
│  │  - Feedback Display                                   │  │
│  │  - Score Dashboard                                    │  │
│  └───────────────────────────────────────────────────────┘  │
│                          ↕                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  State Management (st.session_state)                  │  │
│  │  - Interview configuration                            │  │
│  │  - Conversation history                               │  │
│  │  - Feedback records                                   │  │
│  │  - Score records                                      │  │
│  └───────────────────────────────────────────────────────┘  │
│                          ↕                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  AutoGen Orchestration Layer                          │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │  │
│  │  │ Interviewer │  │   Coach     │  │   Score     │  │  │
│  │  │   Agent     │  │   Agent     │  │   Agent     │  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  │  │
│  │         ↕                                              │  │
│  │  ┌─────────────┐                                      │  │
│  │  │ User Proxy  │                                      │  │
│  │  │   Agent     │                                      │  │
│  │  └─────────────┘                                      │  │
│  └───────────────────────────────────────────────────────┘  │
│                          ↕                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  OpenAI API Integration                               │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
Architecture Principles:

Single Streamlit app: All logic in one coherent application
Session-based state: Everything stored in st.session_state
AutoGen orchestration: Agents communicate through GroupChat
Synchronous flow: User waits for complete agent responses
Extensible design: Clear separation for future backend migration


2. Agent Responsibilities & Communication Flow
Agent Definitions
1️⃣ AI Interviewer Agent
Role: Interview orchestrator and question master

Responsibilities:

Initialize interview with role/domain context
Ask contextually appropriate questions
Control difficulty progression (easy → medium → hard)
Track interview progress (question count)
Explicitly end interview after N questions
Maintain professional, encouraging tone


Output Format:

  QUESTION [X/Y]: <question text>
  DIFFICULTY: <easy|medium|hard>
  CONTEXT: <optional context or scenario>
2️⃣ User Proxy Agent
Role: Human-AI bridge

Responsibilities:

Relay user text input to agent system
No autonomous responses
Act as termination controller
Handle user's "skip" or "quit" commands


Configuration:

human_input_mode="ALWAYS" (conceptually, handled by Streamlit)
max_consecutive_auto_reply=0



3️⃣ Coach Agent
Role: Feedback and improvement advisor

Responsibilities:

Analyze answer quality, depth, and clarity
Provide structured feedback immediately after each answer
Identify strengths and gaps
Suggest specific improvements
Maintain supportive, constructive tone


Output Format (structured text):

  FEEDBACK:
  ✓ Strengths: <list>
  ⚠ Gaps: <list>
  💡 Suggestions: <list>
4️⃣ Score Agent
Role: Objective evaluator

Responsibilities:

Score each answer on 0-10 scale
Evaluate: Technical Accuracy, Communication, Depth, Relevance
Output machine-readable JSON
Calculate cumulative scores


Output Format (JSON):

json  {
    "question_id": 1,
    "technical_accuracy": 8,
    "communication": 7,
    "depth": 6,
    "relevance": 9,
    "total_score": 7.5,
    "justification": "..."
  }
```

### Communication Flow Diagram
```
User Action → Streamlit UI
                ↓
         Update session_state
                ↓
    Trigger AutoGen GroupChat
                ↓
    ┌───────────────────────────┐
    │   Interviewer Agent       │
    │   Posts next question     │
    └───────────────────────────┘
                ↓
         Display in UI
                ↓
         User types answer
                ↓
    ┌───────────────────────────┐
    │   User Proxy Agent        │
    │   Submits answer          │
    └───────────────────────────┘
                ↓
         Parallel Processing
    ┌─────────────┬─────────────┐
    │ Coach Agent │ Score Agent │
    │ Analyzes    │ Evaluates   │
    │ Provides    │ Scores in   │
    │ Feedback    │ JSON        │
    └─────────────┴─────────────┘
                ↓
    Store feedback & scores in session_state
                ↓
         Display in UI panels
                ↓
    Interviewer posts next question
    (or ends interview)
Agent Orchestration Strategy
GroupChat Configuration:

Speaker Selection: round_robin with custom speaker transitions
Transition Rules:

Interviewer → User Proxy (question asked)
User Proxy → Coach + Score (answer submitted)
Coach + Score → Interviewer (feedback/scores complete)



Max Rounds: total_questions * 3 (question → answer → feedback cycle)