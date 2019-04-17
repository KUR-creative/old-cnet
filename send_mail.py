import smtplib, ssl

port = 465  # For SSL
smtp_server = "smtp.gmail.com"
sender_email = input("Type sender address and press enter: ")     # Enter your address
receiver_email = input("Type receiver address and press enter: ") # Enter receiver address
password = input("Type password of sender and press enter: ")
message = """\
Subject: Running ended!

If you doesn't have end running, See log files!"""

# test
context = ssl.create_default_context()
with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
    test_msg = """\
Subject: Mailing test

Is mail sended correctly?
If that, your input sender&receiver email is correct.
Type 'yes' to terminal """
    server.login(sender_email, password)
    server.sendmail(sender_email, receiver_email, test_msg)

yes_or = input("Test mail received? (yes/no): ")
if yes_or != 'yes':
    print('Maybe wrong sender/receiver email address.')
    exit()

def send_mail(string=''):
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message + '\n\n' + string)
