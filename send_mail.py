import smtplib, ssl

port = 465  # For SSL
smtp_server = "smtp.gmail.com"
sender_email = None #input("Type sender address and press enter: ")     # Enter your address
receiver_email = None #input("Type receiver address and press enter: ") # Enter receiver address
password = None #input("Type password of sender and press enter: ")
message = """\
Subject: Running ended!

If you doesn't have end running, See log files!"""

test_msg = """\
Subject: Mailing test

Is mail sended correctly?
If that, your input sender&receiver email is correct."""

def test():
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, test_msg)

def send(string=''):
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message + '\n\n' + string)
