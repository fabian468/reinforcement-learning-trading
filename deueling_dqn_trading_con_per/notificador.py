# -*- coding: utf-8 -*-
"""
Created on Sat May 17 02:27:03 2025

@author: fabia
"""
from twilio.rest import Client
from dotenv import load_dotenv
import os

# Cargar variables del archivo .env
load_dotenv()

ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
WHATSAPP_FROM = os.getenv("WHATSAPP_FROM")
WHATSAPP_TO = os.getenv("WHATSAPP_TO")

def enviar_alerta(mensaje):
    client = Client(ACCOUNT_SID, AUTH_TOKEN)
    client.messages.create(
        from_=WHATSAPP_FROM,
        body=mensaje,
        to=WHATSAPP_TO
    )

