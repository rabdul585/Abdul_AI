import smtplib
import os
from email.message import EmailMessage
from typing import Dict, Any

class NotificationAgent:
    def __init__(self, smtp_config: Dict[str, Any] = None):
        # We'll load from env directly for simplicity, or use passed config
        self.smtp_server = os.getenv("SMTP_SERVER")
        self.smtp_port = int(os.getenv("SMTP_PORT", 587))
        self.smtp_username = os.getenv("SMTP_USERNAME")
        self.smtp_password = os.getenv("SMTP_password") # Note: User env might have lowercase 'password' based on test_email.py? Checking test_email.py it uses SMTP_PASSWORD. Let's stick to standard env vars.
        
        # Fallback to env if not passed
        if not self.smtp_username:
             self.smtp_username = os.getenv("SMTP_USERNAME")
        if not self.smtp_password:
             self.smtp_password = os.getenv("SMTP_PASSWORD")

    def send_notification(self, ticket_id: str, status: str, details: Dict[str, Any], user_email: str = None):
        """
        Sends a notification email via SMTP.
        """
        subject = f"Ticket {ticket_id}: {status}"
        
        body = f"""
        Ticket Update
        -------------
        ID: {ticket_id}
        Status: {status}
        
        Details:
        Category: {details.get('category', 'N/A')}
        User Query: {details.get('query', 'N/A')}
        Resolution/Action: {details.get('action', 'N/A')}
        
        Timestamp: {details.get('timestamp', 'N/A')}
        """
        
        recipient = user_email if user_email else self.smtp_username # Default to self if no user email
        
        try:
            msg = EmailMessage()
            msg["From"] = self.smtp_username
            msg["To"] = recipient
            msg["Subject"] = subject
            msg.set_content(body)

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)
            
            print(f"--- EMAIL SENT to {recipient} ---")
            return {"status": "sent", "recipient": recipient}
            
        except Exception as e:
            print(f"--- EMAIL FAILED: {e} ---")
            return {"status": "failed", "error": str(e)}
