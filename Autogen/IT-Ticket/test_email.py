import os
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv

load_dotenv()

msg = EmailMessage()
msg["From"] = os.getenv("SMTP_USERNAME")
msg["To"] = os.getenv("SMTP_USERNAME")
msg["Subject"] = "SMTP Test - AI IT Support"
msg.set_content("If you received this email, SMTP is configured correctly.")

with smtplib.SMTP(os.getenv("SMTP_SERVER"), int(os.getenv("SMTP_PORT"))) as server:
    server.starttls()
    server.login(
        os.getenv("SMTP_USERNAME"),
        os.getenv("SMTP_PASSWORD")
    )
    server.send_message(msg)

print("✅ Email sent successfully")
