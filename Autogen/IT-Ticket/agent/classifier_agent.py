import autogen
from typing import Dict, Any

class ClassifierAgent:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.config_list = [
            {
                "model": model,
                "api_key": api_key,
            }
        ]
        
        self.agent = autogen.AssistantAgent(
            name="classifier_agent",
            llm_config={
                "config_list": self.config_list,
                "temperature": 0.7, # Increased creativity/flexibility as requested
            },
            system_message="""You are an IT Support Ticket Classifier.
            Your ONLY job is to classify the user's input into exactly ONE of the following categories:
            - Network
            - Software
            - Access
            - Hardware
            - Performance
            - Security
            - Unknown

            Output ONLY the category name. Do not add any punctuation or extra text.
            """
        )
        
        self.user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config=False,
        )

    def classify(self, user_input: str) -> str:
        """
        Classifies the user input into a predefined category.
        """
        # Initiate chat to get classification
        self.user_proxy.initiate_chat(
            self.agent,
            message=f"Classify this issue: {user_input}",
            summary_method="last_msg"
        )
        
        # Extract the last message content which should be the category
        last_message = self.user_proxy.last_message()["content"]
        
        # Clean up the response just in case
        category = last_message.strip().replace(".", "")
        
        valid_categories = ["Network", "Software", "Access", "Hardware", "Performance", "Security", "Unknown"]
        
        if category not in valid_categories:
            return "Unknown"
            
        return category