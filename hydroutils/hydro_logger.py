import smtplib
import ssl


# -------------------------------------------------- notification tools--------------------------------------------
def send_email(subject, text, receiver="hust2014owen@gmail.com"):
    sender = "hydro.wyouyang@gmail.com"
    password = "D4VEFya3UQxGR3z"
    context = ssl.create_default_context()
    msg = f"Subject: {subject}\n\n{text}"
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender, password)
        server.sendmail(from_addr=sender, to_addrs=receiver, msg=msg)
