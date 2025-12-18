"""
AI Interviewer - Streamlit MVP
A complete AI-powered interview system using AutoGen agents

Run: streamlit run app.py
"""

import streamlit as st
import os
from dotenv import load_dotenv
import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

# Load environment variables
load_dotenv()

# Check for required packages
try:
    import autogen
except ImportError:
    st.error("⚠️ AutoGen not installed. Run: pip install pyautogen")
    st.stop()

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class InterviewConfig:
    """Interview configuration state"""
    role: str = ""
    domain: str = ""
    total_questions: int = 5
    started: bool = False
    completed: bool = False

@dataclass
class QuestionRecord:
    """Record of a single question-answer interaction"""
    question_id: int
    question_text: str
    difficulty: str
    answer: str
    feedback: str
    scores: Dict
    timestamp: str

# ============================================================================
# AUTOGEN CONFIGURATION
# ============================================================================

def create_llm_config() -> Dict[str, Any]:
    """Create LLM configuration for all agents"""
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        st.error("⚠️ OPENAI_API_KEY not found in environment variables!")
        st.info("Create a .env file with: OPENAI_API_KEY=your_key_here")
        st.stop()
    
    return {
        "config_list": [
            {
                "model": "gpt-4.1-mini",
                "api_key": api_key,
                "temperature": 0.7,
            }
        ],
        "timeout": 120,
        "cache_seed": None,
    }

def create_interviewer_agent(llm_config: Dict, total_questions: int, role: str, domain: str) -> autogen.AssistantAgent:
    """AI Interviewer Agent"""
    system_message = f"""You are an expert technical interviewer conducting a structured interview for a {role} position in {domain}.

Your responsibilities:
1. Ask ONE question at a time, progressing from easy → medium → hard
2. Label each question: QUESTION [X/{total_questions}] and DIFFICULTY: [easy/medium/hard]
3. After receiving feedback and scores from Coach and Scorer agents, ask the next question
4. After {total_questions} questions, say "INTERVIEW_COMPLETE" and provide a brief summary
5. Maintain a professional, encouraging tone

Format each question like this:
QUESTION [1/{total_questions}]
DIFFICULTY: easy
[Your question here]

Focus on behavioral and technical questions appropriate to the role and domain.
"""
    
    return autogen.AssistantAgent(
        name="Interviewer",
        system_message=system_message,
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

def create_coach_agent(llm_config: Dict) -> autogen.AssistantAgent:
    """Coach/Feedback Agent"""
    system_message = """You are an interview coach providing constructive feedback.

After each candidate answer, provide structured feedback using this format:

FEEDBACK:
✓ Strengths:
- [Point 1]
- [Point 2]

⚠ Gaps:
- [Point 1]
- [Point 2]

💡 Suggestions:
- [Point 1]
- [Point 2]

Be supportive but honest. Keep feedback concise (2-4 bullet points per section).
"""
    
    return autogen.AssistantAgent(
        name="Coach",
        system_message=system_message,
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

def create_score_agent(llm_config: Dict) -> autogen.AssistantAgent:
    """Scoring Agent"""
    system_message = """You are an objective interview evaluator.

Score each answer on these dimensions (0-10 scale):
- technical_accuracy: Correctness and precision
- communication: Clarity and structure
- depth: Level of detail and insight
- relevance: Alignment with question

Output ONLY valid JSON in this exact format (no other text):
{
    "question_id": 1,
    "technical_accuracy": 8,
    "communication": 7,
    "depth": 6,
    "relevance": 9,
    "total_score": 7.5,
    "justification": "Brief explanation here"
}

Calculate total_score as the average of the four dimensions.
"""
    
    return autogen.AssistantAgent(
        name="Scorer",
        system_message=system_message,
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

# ============================================================================
# INTERVIEW ORCHESTRATOR
# ============================================================================

class InterviewOrchestrator:
    """Manages AutoGen agent interactions for interview flow"""
    
    def __init__(self, role: str, domain: str, total_questions: int = 5):
        self.role = role
        self.domain = domain
        self.total_questions = total_questions
        self.current_question = 0
        
        # Initialize agents
        llm_config = create_llm_config()
        self.interviewer = create_interviewer_agent(llm_config, total_questions, role, domain)
        self.coach = create_coach_agent(llm_config)
        self.scorer = create_score_agent(llm_config)
        
        # User proxy (represents candidate)
        self.user_proxy = autogen.UserProxyAgent(
            name="Candidate",
            system_message="You are the interview candidate.",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config=False,
        )
        
        # Create group chat with specific speaker transitions
        self.group_chat = autogen.GroupChat(
            agents=[self.interviewer, self.user_proxy, self.coach, self.scorer],
            messages=[],
            max_round=total_questions * 6,
            speaker_selection_method="auto",
            allow_repeat_speaker=False,
        )
        
        self.manager = autogen.GroupChatManager(
            groupchat=self.group_chat,
            llm_config=llm_config,
        )
    
    def start_interview(self) -> str:
        """Initialize interview and get first question"""
        init_message = f"""Begin the interview now. This is for a {self.role} position in {self.domain}.
Total questions: {self.total_questions}.
Ask the first question immediately."""
        
        # Start conversation
        self.user_proxy.initiate_chat(
            self.manager,
            message=init_message,
            clear_history=True,
        )
        
        return self._extract_last_question()
    
    def submit_answer(self, answer: str) -> Dict[str, Any]:
        """Submit user answer and get feedback + scores"""
        self.current_question += 1
        
        # Send answer
        self.user_proxy.send(
            message=f"CANDIDATE ANSWER: {answer}",
            recipient=self.manager,
            request_reply=True,
        )
        
        # Wait for coach and scorer responses
        # The group chat will automatically route to coach and scorer
        
        # Extract results
        feedback = self._extract_coach_feedback()
        scores = self._extract_scores()
        
        # Check completion before getting next question
        interview_complete = self._check_completion()
        
        if not interview_complete:
            # Request next question from interviewer
            next_question = self._extract_last_question()
        else:
            next_question = None
        
        return {
            "feedback": feedback,
            "scores": scores,
            "next_question": next_question,
            "interview_complete": interview_complete,
        }
    
    def _extract_last_question(self) -> Optional[str]:
        """Extract most recent question from conversation"""
        for msg in reversed(self.group_chat.messages):
            if msg.get("name") == "Interviewer":
                content = msg.get("content", "")
                if "QUESTION" in content.upper():
                    return content
        return "No question available"
    
    def _extract_coach_feedback(self) -> str:
        """Extract coach feedback from conversation"""
        for msg in reversed(self.group_chat.messages):
            if msg.get("name") == "Coach":
                content = msg.get("content", "")
                if "FEEDBACK" in content or "Strengths" in content:
                    return content
        return "No feedback available yet"
    
    def _extract_scores(self) -> Dict:
        """Extract and parse JSON scores"""
        for msg in reversed(self.group_chat.messages):
            if msg.get("name") == "Scorer":
                content = msg.get("content", "")
                try:
                    # Try to extract JSON
                    json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
                    if json_match:
                        scores = json.loads(json_match.group())
                        return scores
                except (json.JSONDecodeError, AttributeError):
                    pass
        
        # Return default scores if parsing fails
        return {
            "question_id": self.current_question,
            "technical_accuracy": 5,
            "communication": 5,
            "depth": 5,
            "relevance": 5,
            "total_score": 5.0,
            "justification": "Score parsing unavailable"
        }
    
    def _check_completion(self) -> bool:
        """Check if interview is complete"""
        # Check for explicit completion message
        for msg in reversed(self.group_chat.messages):
            content = msg.get("content", "")
            if "INTERVIEW_COMPLETE" in content.upper():
                return True
        
        # Check if we've reached question limit
        return self.current_question >= self.total_questions

# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================

def initialize_session_state():
    """Initialize all session state variables"""
    if 'config' not in st.session_state:
        st.session_state.config = InterviewConfig()
    
    if 'orchestrator' not in st.session_state:
        st.session_state.orchestrator = None
    
    if 'current_question' not in st.session_state:
        st.session_state.current_question = None
    
    if 'current_answer' not in st.session_state:
        st.session_state.current_answer = ""
    
    if 'records' not in st.session_state:
        st.session_state.records: List[QuestionRecord] = []
    
    if 'loading' not in st.session_state:
        st.session_state.loading = False
    
    if 'error_message' not in st.session_state:
        st.session_state.error_message = None

def reset_interview():
    """Reset all interview state"""
    st.session_state.config = InterviewConfig()
    st.session_state.orchestrator = None
    st.session_state.current_question = None
    st.session_state.current_answer = ""
    st.session_state.records = []
    st.session_state.loading = False
    st.session_state.error_message = None

def add_question_record(question: str, difficulty: str, answer: str, 
                       feedback: str, scores: Dict):
    """Add a complete question-answer record"""
    record = QuestionRecord(
        question_id=len(st.session_state.records) + 1,
        question_text=question,
        difficulty=difficulty,
        answer=answer,
        feedback=feedback,
        scores=scores,
        timestamp=datetime.now().isoformat()
    )
    st.session_state.records.append(record)

# ============================================================================
# UI COMPONENTS
# ============================================================================

def apply_custom_styles():
    """Apply custom CSS styling"""
    st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
    }
    .question-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .feedback-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #2ecc71;
    }
    .score-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
    }
    </style>
    """, unsafe_allow_html=True)

def render_config_screen():
    """Interview configuration and start screen"""
    
    st.header("🎯 Configure Your Interview")
    st.markdown("*Set up your AI-powered interview session*")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Role & Domain")
        
        role = st.selectbox(
            "Select Role",
            ["Software Engineer", "Data Scientist", "Product Manager", 
             "DevOps Engineer", "Frontend Developer", "Backend Developer",
             "Full Stack Developer", "ML Engineer"],
            key="role_select"
        )
        
        domain = st.selectbox(
            "Select Domain",
            ["Web Development", "Machine Learning", "System Design",
             "Cloud Architecture", "Data Engineering", "Mobile Development",
             "API Development", "Database Design"],
            key="domain_select"
        )
    
    with col2:
        st.subheader("⚙️ Interview Settings")
        
        total_questions = st.slider(
            "Number of Questions",
            min_value=3,
            max_value=10,
            value=5,
            key="questions_slider"
        )
        
        st.info(f"""
        **Interview Structure:**
        - 📊 {total_questions} questions total
        - 📈 Progressive difficulty (easy → hard)
        - 💬 Real-time feedback after each answer
        - 🎯 Comprehensive scoring
        """)
    
    st.divider()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🚀 Start Interview", type="primary", use_container_width=True):
            # Update config
            st.session_state.config.role = role
            st.session_state.config.domain = domain
            st.session_state.config.total_questions = total_questions
            st.session_state.config.started = True
            
            # Initialize orchestrator
            with st.spinner("🤖 Initializing AI interviewers..."):
                try:
                    st.session_state.orchestrator = InterviewOrchestrator(
                        role=role,
                        domain=domain,
                        total_questions=total_questions
                    )
                    
                    # Get first question
                    first_question = st.session_state.orchestrator.start_interview()
                    st.session_state.current_question = first_question
                    
                    st.success("✅ Interview started successfully!")
                    
                except Exception as e:
                    st.error(f"Error starting interview: {str(e)}")
                    st.session_state.config.started = False
                    return
            
            st.rerun()

def render_interview_screen():
    """Main interview interface"""
    
    # Header with progress
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.header("💼 Interview in Progress")
        st.caption(f"Role: {st.session_state.config.role} | Domain: {st.session_state.config.domain}")
    
    with col2:
        progress = len(st.session_state.records)
        total = st.session_state.config.total_questions
        st.metric("Progress", f"{progress}/{total}")
        st.progress(progress / total if total > 0 else 0)
    
    with col3:
        if st.button("🔄 Restart", type="secondary"):
            if st.session_state.records:
                st.warning("This will reset your progress. Click again to confirm.")
            reset_interview()
            st.rerun()
    
    st.divider()
    
    # Two-column layout
    col_main, col_sidebar = st.columns([2, 1])
    
    with col_main:
        render_question_answer_panel()
    
    with col_sidebar:
        render_feedback_scoring_panel()

def render_question_answer_panel():
    """Question display and answer input"""
    
    st.subheader("💬 Current Question")
    
    # Display current question
    if st.session_state.current_question:
        st.markdown(f"""
        <div class="question-box">
        {st.session_state.current_question.replace('QUESTION', '**QUESTION**').replace('DIFFICULTY:', '**DIFFICULTY:**')}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Waiting for question...")
    
    st.divider()
    
    # Answer input
    st.subheader("✍️ Your Answer")
    
    answer = st.text_area(
        "Type your response here...",
        value=st.session_state.current_answer,
        height=200,
        key="answer_input",
        placeholder="Share your thoughts, experiences, or technical approach...",
        disabled=st.session_state.loading
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col2:
        if st.button("⏭️ Skip", type="secondary", disabled=st.session_state.loading):
            st.warning("Skipping questions will affect your score")
    
    with col3:
        submit_disabled = not answer.strip() or st.session_state.loading
        
        if st.button("📤 Submit", type="primary", disabled=submit_disabled):
            st.session_state.loading = True
            st.session_state.current_answer = answer
            st.rerun()
    
    # Process submission
    if st.session_state.loading and st.session_state.current_answer.strip():
        with st.spinner("🤖 AI agents are analyzing your answer..."):
            process_answer_submission(st.session_state.current_answer)
    
    # Display conversation history
    if st.session_state.records:
        st.divider()
        render_conversation_history()

def process_answer_submission(answer: str):
    """Process answer through AutoGen orchestrator"""
    
    try:
        # Submit to orchestrator
        result = st.session_state.orchestrator.submit_answer(answer)
        
        # Extract components
        feedback = result.get("feedback", "No feedback available")
        scores = result.get("scores", {})
        next_question = result.get("next_question")
        is_complete = result.get("interview_complete", False)
        
        # Parse current question for metadata
        current_q_text = st.session_state.current_question
        difficulty = "medium"
        
        if "DIFFICULTY:" in current_q_text:
            try:
                difficulty = current_q_text.split("DIFFICULTY:")[1].split("\n")[0].strip().lower()
            except:
                difficulty = "medium"
        
        # Save record
        add_question_record(
            question=current_q_text,
            difficulty=difficulty,
            answer=answer,
            feedback=feedback,
            scores=scores
        )
        
        # Update state
        if is_complete:
            st.session_state.config.completed = True
        else:
            st.session_state.current_question = next_question
        
        st.session_state.current_answer = ""
        st.session_state.loading = False
        
        st.success("✅ Answer submitted successfully!")
        st.rerun()
        
    except Exception as e:
        st.error(f"⚠️ Error processing answer: {str(e)}")
        st.session_state.loading = False
        st.session_state.error_message = str(e)

def render_conversation_history():
    """Display previous Q&A pairs"""
    
    st.subheader("📜 Previous Questions")
    
    for record in st.session_state.records:
        with st.expander(f"Q{record.question_id}: {record.difficulty.upper()} - Score: {record.scores.get('total_score', 'N/A'):.1f}/10"):
            st.markdown(f"**Question:**")
            st.info(record.question_text)
            st.markdown(f"**Your Answer:**")
            st.write(record.answer)
            st.caption(f"⏰ {record.timestamp}")

def render_feedback_scoring_panel():
    """Real-time feedback and scoring display"""
    
    st.subheader("📊 Latest Feedback")
    
    if not st.session_state.records:
        st.info("💡 Complete your first answer to see feedback and scores here!")
        return
    
    # Show most recent feedback
    latest_record = st.session_state.records[-1]
    
    # Feedback card
    st.markdown(f"""
    <div class="feedback-box">
    {latest_record.feedback.replace('✓', '✅').replace('⚠', '⚠️').replace('💡', '💡')}
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Scoring card
    st.subheader("🎯 Latest Scores")
    
    scores = latest_record.scores
    
    if scores:
        # Score metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("🎓 Technical", f"{scores.get('technical_accuracy', 0):.1f}/10")
            st.metric("📊 Depth", f"{scores.get('depth', 0):.1f}/10")
        
        with col2:
            st.metric("💬 Communication", f"{scores.get('communication', 0):.1f}/10")
            st.metric("🎯 Relevance", f"{scores.get('relevance', 0):.1f}/10")
        
        st.divider()
        
        # Total score
        total = scores.get('total_score', 0)
        st.metric("⭐ Total Score", f"{total:.1f}/10")
        st.progress(total / 10)
        
        # Justification
        if 'justification' in scores:
            with st.expander("📝 Score Justification"):
                st.write(scores['justification'])
    
    # Overall statistics
    if len(st.session_state.records) > 1:
        st.divider()
        render_overall_stats()

def render_overall_stats():
    """Display cumulative statistics"""
    
    st.subheader("📈 Overall Performance")
    
    records = st.session_state.records
    
    # Calculate averages
    avg_technical = sum(r.scores.get('technical_accuracy', 0) for r in records) / len(records)
    avg_communication = sum(r.scores.get('communication', 0) for r in records) / len(records)
    avg_depth = sum(r.scores.get('depth', 0) for r in records) / len(records)
    avg_total = sum(r.scores.get('total_score', 0) for r in records) / len(records)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Avg Technical", f"{avg_technical:.1f}")
        st.metric("Avg Depth", f"{avg_depth:.1f}")
    
    with col2:
        st.metric("Avg Communication", f"{avg_communication:.1f}")
        st.metric("Avg Overall", f"{avg_total:.1f}")

def render_summary_screen():
    """Final interview summary and report"""
    
    st.header("🎉 Interview Complete!")
    st.success("Congratulations! You've completed the interview. Here's your comprehensive report.")
    
    st.divider()
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    records = st.session_state.records
    avg_total = sum(r.scores.get('total_score', 0) for r in records) / len(records) if records else 0
    
    with col1:
        st.metric("📝 Questions", len(records))
    
    with col2:
        st.metric("⭐ Avg Score", f"{avg_total:.1f}/10")
    
    with col3:
        st.metric("👔 Role", st.session_state.config.role)
    
    with col4:
        st.metric("🎯 Domain", st.session_state.config.domain)
    
    st.divider()
    
    # Detailed breakdown
    tabs = st.tabs(["📊 Score Analysis", "💬 Full Transcript", "🎓 Key Takeaways"])
    
    with tabs[0]:
        render_score_analysis()
    
    with tabs[1]:
        render_full_transcript()
    
    with tabs[2]:
        render_key_takeaways()
    
    # Actions
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Start New Interview", type="primary", use_container_width=True):
            reset_interview()
            st.rerun()
    
    with col2:
        if st.button("📥 Download Report", type="secondary", use_container_width=True):
            generate_report()

def render_score_analysis():
    """Detailed score breakdown"""
    
    st.subheader("Performance Analysis")
    
    records = st.session_state.records
    
    if not records:
        st.warning("No data available")
        return
    
    # Create score dataframe
    import pandas as pd
    
    df = pd.DataFrame([
        {
            'Question': f"Q{r.question_id}",
            'Technical': r.scores.get('technical_accuracy', 0),
            'Communication': r.scores.get('communication', 0),
            'Depth': r.scores.get('depth', 0),
            'Relevance': r.scores.get('relevance', 0),
            'Total': r.scores.get('total_score', 0),
            'Difficulty': r.difficulty
        }
        for r in records
    ])
    
    # Display chart
    st.line_chart(df.set_index('Question')[['Technical', 'Communication', 'Depth', 'Relevance']])
    
    # Display table
    st.dataframe(df, use_container_width=True)
    
    # Summary stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Best Score", f"{df['Total'].max():.1f}/10")
    
    with col2:
        st.metric("Average", f"{df['Total'].mean():.1f}/10")
    
    with col3:
        st.metric("Lowest Score", f"{df['Total'].min():.1f}/10")

def render_full_transcript():
    """Complete interview transcript"""
    
    st.subheader("Complete Interview Transcript")
    
    for record in st.session_state.records:
        st.markdown(f"### Question {record.question_id} ({record.difficulty.upper()})")
        
        with st.container(border=True):
            st.markdown("**Question:**")
            st.info(record.question_text)
            
            st.markdown("**Your Answer:**")
            st.write(record.answer)
            
            st.markdown("**Feedback:**")
            st.success(record.feedback)
            
            st.markdown("**Scores:**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Tech", record.scores.get('technical_accuracy', 0))
            with col2:
                st.metric("Comm", record.scores.get('communication', 0))
            with col3:
                st.metric("Depth", record.scores.get('depth', 0))
            with col4:
                st.metric("Total", record.scores.get('total_score', 0))
        
        st.divider()

def render_key_takeaways():
    """Key insights and action items"""
    
    st.subheader("Key Takeaways")
    
    records = st.session_state.records
    
    # Calculate averages
    avg_scores = {
        'technical': sum(r.scores.get('technical_accuracy', 0) for r in records) / len(records),
        'communication': sum(r.scores.get('communication', 0) for r in records) / len(records),
        'depth': sum(r.scores.get('depth', 0) for r in records) / len(records),
        'relevance': sum(r.scores.get('relevance', 0) for r in records) / len(records),
    }
    
    # Strengths
    st.markdown("### ✅ Strengths")
    strengths = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)[:2]
    for skill, score in strengths:
        st.success(f"**{skill.title()}**: {score:.1f}/10 - Strong performance in this area!")
    
    # Areas for improvement
    st.markdown("### 🎯 Areas for Improvement")
    improvements = sorted(avg_scores.items(), key=lambda x: x[1])[:2]
    for skill, score in improvements:
        st.warning(f"**{skill.title()}**: {score:.1f}/10 - Focus on developing this skill")
    
    # Recommendations
    st.markdown("### 💡 Recommendations")
    st.info("""
    - Practice more technical questions in your domain
    - Work on structuring your answers clearly
    - Provide more specific examples from your experience
    - Review fundamental concepts before your next interview
    """)

def generate_report():
    """Generate downloadable report"""
    
    records = st.session_state.records
    
    # Create text report
    report = f"""
AI INTERVIEWER - INTERVIEW REPORT
================================

Candidate Information:
- Role: {st.session_state.config.role}
- Domain: {st.session_state.config.domain}
- Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

Overall Performance:
- Questions Answered: {len(records)}
- Average Score: {sum(r.scores.get('total_score', 0) for r in records) / len(records):.2f}/10

Detailed Results:
"""
    
    for record in records:
        report += f"""
---
Question {record.question_id} ({record.difficulty.upper()})
Q: {record.question_text}
A: {record.answer}

Scores:
- Technical: {record.scores.get('technical_accuracy', 0)}/10
- Communication: {record.scores.get('communication', 0)}/10
- Depth: {record.scores.get('depth', 0)}/10
- Total: {record.scores.get('total_score', 0)}/10

Feedback:
{record.feedback}
"""
    
    # Create download button
    st.download_button(
        label="📄 Download Full Report",
        data=report,
        file_name=f"interview_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain"
    )

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point"""
    
    # Page configuration
    st.set_page_config(
        page_title="AI Interviewer",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Apply custom styling
    apply_custom_styles()
    
    # Main title
    st.title("🎯 AI Interviewer")
    st.markdown("*Practice interviews with AI-powered feedback and scoring*")
    st.divider()
    
    # Routing logic based on interview state
    if not st.session_state.config.started:
        render_config_screen()
    elif st.session_state.config.completed:
        render_summary_screen()
    else:
        render_interview_screen()

if __name__ == "__main__":
    main()