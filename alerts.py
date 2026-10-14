"""
alerts.py
---------
Twilio SMS alerts for accident detection, with a per-direction cooldown so a
single incident doesn't spam messages. Sending runs in a background thread so
it never blocks the detection loop.

Credentials come from config.py (which reads env vars first). If Twilio isn't
installed or credentials are missing, alerts are logged to the console instead
so the rest of the system keeps working.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Dict

import config

_LAST_SENT: Dict[str, float] = {}          # accident cooldown per direction
_LAST_CONGESTION: Dict[str, float] = {}    # congestion cooldown per direction
_lock = threading.Lock()

_client = None
_twilio_ok = False

_from_number = config.TWILIO_FROM

if config.TWILIO_ENABLED:
    try:
        from twilio.rest import Client
        if config.TWILIO_SID and config.TWILIO_AUTH:
            _client = Client(config.TWILIO_SID, config.TWILIO_AUTH)
            _twilio_ok = True
            print("[alerts] Twilio client initialised")

            # Ignore obvious placeholder / empty FROM values.
            if (not _from_number or "YOUR" in _from_number.upper()
                    or not _from_number.startswith("+")):
                _from_number = None

            # Auto-detect a real Twilio number on the account (best FROM).
            try:
                owned = _client.incoming_phone_numbers.list(limit=5)
                if owned:
                    _from_number = owned[0].phone_number
                    print(f"[alerts] using Twilio number {_from_number}")
                elif _from_number is None:
                    print("[alerts] WARNING: your Twilio account owns no phone "
                          "number. Buy one in the Console (Phone Numbers -> "
                          "Buy a number) to enable SMS. Alerts will be logged.")
            except Exception as exc:
                print(f"[alerts] could not list Twilio numbers: {exc}")
    except Exception as exc:  # twilio not installed or bad creds
        print(f"[alerts] Twilio unavailable ({exc}); SMS will be logged only")


def twilio_available() -> bool:
    """True only if we can actually send (client + a valid FROM number)."""
    return _twilio_ok and _from_number is not None


def status_text() -> str:
    if not _twilio_ok:
        return "Twilio not configured (SMS logged to console)."
    if _from_number is None:
        return ("Twilio connected but NO sender number — buy one in the "
                "Twilio Console to enable SMS.")
    return f"SMS ready (from {_from_number})."


def _send(body: str, tag: str) -> None:
    if not _twilio_ok or _client is None or _from_number is None:
        print(f"[alerts] (SMS not sent - no Twilio sender number) {body}")
        return
    # Trial accounts only accept predefined template names as the body. We log
    # the full descriptive message to the console but send the template so the
    # SMS actually delivers. After upgrading Twilio, set TWILIO_TRIAL_TEMPLATES
    # = False in .env to send the real text.
    send_body = (config.TWILIO_TEMPLATE
                 if config.TWILIO_TRIAL_TEMPLATES else body)
    try:
        msg = _client.messages.create(
            from_=_from_number, to=config.TWILIO_TO, body=send_body)
        print(f"[alerts] SMS sent ({tag}): {msg.sid} | {body}")
    except Exception as exc:
        print(f"[alerts] SMS send failed ({tag}): {exc}")


def send_accident_sms(direction: str, conf: float = 0.0) -> bool:
    """
    Fire an SMS for an accident at `direction`, respecting the cooldown.
    Returns True if a message was dispatched (queued), False if suppressed.
    """
    now = time.time()
    with _lock:
        last = _LAST_SENT.get(direction, 0.0)
        if now - last < config.SMS_COOLDOWN:
            return False
        _LAST_SENT[direction] = now
    body = (f"ACCIDENT detected at {direction} approach "
            f"(confidence {conf:.0%}) at "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. "
            f"Emergency response recommended.")
    threading.Thread(target=_send, args=(body, f"accident/{direction}"),
                     daemon=True).start()
    return True


def send_congestion_sms(direction: str, vehicles: int,
                        green_dir: str = "") -> bool:
    """
    Fire an SMS when `direction` is heavily congested and should get the green.
    Has its own per-direction cooldown so it doesn't spam.
    Returns True if a message was dispatched, False if suppressed.
    """
    now = time.time()
    with _lock:
        last = _LAST_CONGESTION.get(direction, 0.0)
        if now - last < config.CONGESTION_COOLDOWN:
            return False
        _LAST_CONGESTION[direction] = now
    extra = (f" Current green is {green_dir}." if green_dir
             and green_dir != direction else "")
    body = (f"HEAVY TRAFFIC at {direction} approach: {vehicles} vehicles "
            f"queued (>{config.CONGESTION_THRESHOLD}). Recommend switching the "
            f"signal to give {direction} priority.{extra} "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    threading.Thread(target=_send, args=(body, f"congestion/{direction}"),
                     daemon=True).start()
    return True
