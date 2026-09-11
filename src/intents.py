"""
Intent taxonomy and routing policy for the AppleSupport agent.

This taxonomy was NOT invented up front -- it was derived by reading a stratified
sample of ~200 real customer->AppleSupport conversation openers from the
Customer Support on Twitter dataset. See data/labeling_note.md for the full
sampling/labeling methodology and honesty notes about what the raw keyword
counts get wrong.
"""

INTENTS = {
    "software_bug_or_update": {
        "description": (
            "Customer reports something broken, slow, glitchy, or misbehaving, "
            "usually right after an iOS/macOS update (freezing, crashes, battery "
            "drain, the iOS 11.1 autocorrect 'I' bug, UI glitches)."
        ),
        "examples": [
            "My phone hangs constantly since I updated to iOS 11.1, help!",
            "Why does the letter I turn into A? every time I type since the update?",
        ],
    },
    "hardware_defect": {
        "description": (
            "A physical defect in the device itself, not caused by software: "
            "cracked/stained screen, swollen battery, broken buttons, faulty "
            "speaker/mic, defective accessories (cables, headphones)."
        ),
        "examples": [
            "My battery looks swollen after only 6 months, is this normal?",
            "Third pair of Lightning headphones has broken this year.",
        ],
    },
    "account_subscription": {
        "description": (
            "Apple ID / iCloud login issues, Touch ID / Face ID auth failures, "
            "Apple Music / subscription problems, billing/charging disputes, "
            "phishing/scam concerns tied to the account."
        ),
        "examples": [
            "I can't sign into iCloud no matter how many times I reset my password.",
            "I was charged twice for the same Apple Music subscription.",
        ],
    },
    "how_to_question": {
        "description": (
            "A genuine informational question about how to use a feature or "
            "policy -- not a complaint about something broken."
        ),
        "examples": [
            "How do I check my iPhone's battery health?",
            "Can AppleCare+ be transferred if I sell my phone?",
        ],
    },
    "warranty_repair": {
        "description": (
            "Questions or complaints about warranty coverage, Genius Bar "
            "appointments, service center visits, or repair status/quality."
        ),
        "examples": [
            "How do I book a Genius Bar appointment in the UK?",
            "Service center says my cracked screen isn't covered, is that right?",
        ],
    },
    "connectivity_compat": {
        "description": (
            "Bluetooth, Wi-Fi, cellular/SIM, CarPlay, or third-party device "
            "pairing/compatibility issues -- the network or pairing layer, not "
            "a general software crash."
        ),
        "examples": [
            "My Bluetooth turns itself on every time I restart my iPhone.",
            "My phone shows no SIM / no service since this morning.",
        ],
    },
    "general_complaint_venting": {
        "description": (
            "Catch-all for messages with genuinely no specific, actionable "
            "technical detail -- pure frustration/anger with nothing concrete "
            "to diagnose. NOTE: manual review found this is much rarer than a "
            "naive keyword classifier suggests (~2.5% vs an estimated ~15-20%); "
            "most angry-sounding tweets do name a specific real issue."
        ),
        "examples": [
            "You are a crook, Apple.",
            "Check your DMs please.",
        ],
    },
}

# Routing policy: stated explicitly so the "why" is always auditable, not vibes.
ROUTING_POLICY = """
Decide AUTO_HANDLE vs ESCALATE using this policy:

- AUTO_HANDLE if the message is a general how-to/policy question answerable from
  public knowledge with no account access, OR a widely-known bug/behavior change
  that has a published public workaround or explanation article (e.g. the iOS 11.1
  autocorrect bug, or a documented Bluetooth behavior change in iOS 11).
- ESCALATE if resolving it requires: account-specific data (Apple ID, iCloud,
  billing), a serial number / case ID / physical inspection, or an open-ended
  diagnostic back-and-forth (e.g. "which device and iOS version are you on?")
  that can't be answered with a single grounded public reply.
- ESCALATE general_complaint_venting messages: there is not enough detail to
  safely auto-resolve, and a human should gather more information.

Always state a one-sentence reason tied to this policy, not a vague justification.
"""

INTENT_LIST = list(INTENTS.keys())
