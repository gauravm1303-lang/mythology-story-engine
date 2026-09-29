import os
import json
import time
from google import genai


MODEL_NAME = "gemini-3.8-flash"


def load_json(filename):
    with open(filename, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(filename, data):
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def extract_json(text):
    if not text:
        raise ValueError("Editor returned an empty response.")

    text = text.strip()

    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(
            "Editor response did not contain a valid JSON object."
        )

    return json.loads(text[start:end + 1])


def get_episode_plan(season_plan, episode_number):
    for episode in season_plan["episodes"]:
        if episode["episode"] == episode_number:
            return episode

    raise ValueError(
        f"Episode {episode_number} not found in season_1_plan.json"
    )


def build_editor_prompt(
    universe,
    characters,
    episode_plan,
    draft
):
    return f"""
You are the senior screenplay editor and continuity supervisor
for an ORIGINAL Indian mythology-inspired cinematic anime series.

SERIES:
Suryaputra: The Last Legacy

You are NOT writing a completely different episode.

Your job is to POLISH the existing screenplay while preserving
its approved story events.

==================================================
MASTER UNIVERSE
==================================================

{json.dumps(universe, ensure_ascii=False, indent=2)}

==================================================
CHARACTER DATABASE
==================================================

{json.dumps(characters, ensure_ascii=False, indent=2)}

==================================================
APPROVED EPISODE PLAN
==================================================

{json.dumps(episode_plan, ensure_ascii=False, indent=2)}

==================================================
CURRENT DRAFT
==================================================

{json.dumps(draft, ensure_ascii=False, indent=2)}

==================================================
EDITORIAL OBJECTIVES
==================================================

Improve:

- chronology
- continuity
- natural Hindi/Hinglish dialogue
- emotional impact
- pacing
- suspense
- cinematic visual descriptions
- character consistency
- mythology sensitivity
- production usability

Do NOT rewrite the episode into a different story.

Preserve the approved major events.

==================================================
CRITICAL CONTINUITY FIXES
==================================================

1. TIME CONTINUITY

The episode must progress chronologically.

Do not move from Night backward to Evening.

Correct scene times so the episode naturally progresses through
evening, stormy night and pre-dawn where appropriate.

2. AARAV IS NOT KARNA'S REINCARNATION

Never imply that Aarav is literally Karna reborn.

Never state as confirmed fact that Aarav is Karna's descendant.

Never state that divine power belongs to Aarav because of bloodline.

The true Season 1 answer is connected primarily to qualities such
as protection, sacrifice, courage and intent.

Characters may misunderstand the phenomenon when appropriate,
but the screenplay must not accidentally establish a false fact.

3. VASU'S DIALOGUE

Avoid dialogue that definitively says Aarav's responsibility or
power is written in his blood.

Vasu may speak cryptically about responsibility, choices, destiny,
protection or burdens without confirming a bloodline explanation.

4. GENERAL RAHMAN

Rahman must remain morally ambiguous.

Avoid language that definitively identifies Aarav as a literal
divine descendant.

Rahman may refer to:

- solar resonance
- the seal
- the golden phenomenon
- the legacy
- the signal

without claiming Aarav is a divine incarnation.

5. POWER LEVEL

This is Aarav's FIRST supernatural manifestation.

The golden shield must be:

- involuntary
- unstable
- defensive
- shocking
- exhausting

Do not give Aarav controlled offensive powers.

6. MALIK

Do not reveal:

- Malik's complete identity
- Malik's origin
- Malik's full powers
- Malik's complete objective

Only the clue planned for Episode 1 may be established.

7. ACHARYA VEDANT

Vedant may recognize signs of an ancient awakening.

He must NOT already possess every answer.

His dialogue should remain mysterious without spoiling later
episodes.

==================================================
DIALOGUE RULES
==================================================

Dialogue should sound like believable modern Indian conversation.

Use Hindi primarily in Devanagari.

Natural English words may appear where appropriate.

Avoid overly theatrical dialogue unless the character and moment
justify it.

Family conversations should feel warm and believable.

Emotional scenes should feel human rather than melodramatic.

Do not overuse exposition.

==================================================
MYTHOLOGY RULES
==================================================

Treat Indian mythology respectfully.

The modern story, Unit Seven, Divya Resonance, Malik's operations
and the fictional surviving artifact system are fictional.

Do not present invented series lore as established scripture or
historical fact.

Do not change established mythology merely to create spectacle.

==================================================
VISUAL RULES
==================================================

Maintain the exact recurring character identities defined in
characters.json.

Descriptions should remain suitable for later AI anime generation.

Do not mention celebrities.

Do not imitate an existing anime franchise.

Use an original premium cinematic Indian mythology-inspired
anime aesthetic.

Keep visual descriptions specific but not unnecessarily bloated.

==================================================
VIOLENCE RULES
==================================================

The family tragedy is essential to Episode 1.

Preserve the deaths required by the approved episode plan.

Keep the scene emotionally powerful without excessive gore.

Avoid unnecessary graphic descriptions of wounds or blood.

==================================================
RUNTIME
==================================================

Target total runtime:
600 to 900 seconds.

Ideal:
approximately 11 to 13 minutes.

Keep approximately 14 to 22 scenes.

Do not add filler simply to increase duration.

==================================================
OUTPUT
==================================================

Return ONLY valid JSON.

No markdown.

No ```json fences.

No explanation outside JSON.

Preserve this structure:

{{
  "episode_number": 1,
  "title": "The Golden Shield",
  "target_duration_minutes": 12,
  "opening_hook": "Opening hook",

  "scenes": [
    {{
      "scene_number": 1,
      "location": "Location",
      "time": "Time",

      "characters": [
        "Character"
      ],

      "visual_description": "Visual description",

      "camera_direction": "Camera direction",

      "atmosphere": "Atmosphere",

      "dialogues": [
        {{
          "character": "Character",
          "text": "Dialogue",
          "emotion": "Emotion"
        }}
      ],

      "sound_effects": [
        "Sound"
      ],

      "background_music": "Original score description",

      "estimated_duration_seconds": 40
    }}
  ],

  "ending_cliffhanger": "Cliffhanger",

  "episode_summary": "Continuity summary",

  "continuity_updates": [
    "Important continuity fact"
  ]
}}

Before returning the JSON, internally verify:

- chronology is logical
- episode plan is preserved
- no bloodline claim has been accidentally established
- Aarav is not described as Karna's reincarnation
- future mysteries remain hidden
- future powers remain locked
- character designs remain consistent
- family deaths remain permanent
- dialogue sounds natural
- total duration remains 10-15 minutes
- JSON is valid

Now polish the screenplay.
"""


def validate_final_episode(data, episode_number):
    required_keys = [
        "episode_number",
        "title",
        "scenes",
        "ending_cliffhanger",
        "episode_summary",
        "continuity_updates"
    ]

    for key in required_keys:
        if key not in data:
            raise ValueError(
                f"Polished episode missing required key: {key}"
            )

    if not isinstance(data["scenes"], list):
        raise ValueError("scenes must be a list.")

    if not data["scenes"]:
        raise ValueError("Polished episode contains no scenes.")

    data["episode_number"] = episode_number

    total_duration = 0

    for scene in data["scenes"]:
        duration = scene.get(
            "estimated_duration_seconds",
            0
        )

        if isinstance(duration, (int, float)):
            total_duration += duration

    data["calculated_duration_seconds"] = total_duration
    data["calculated_duration_minutes"] = round(
        total_duration / 60,
        2
    )

    return data


def polish_episode(episode_number):
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing from GitHub Secrets."
        )

    draft_file = (
        f"episode_{episode_number:02d}_draft.json"
    )

    if not os.path.exists(draft_file):
        raise FileNotFoundError(
            f"{draft_file} was not found. "
            "Generate the draft first."
        )

    print("Loading universe...")
    universe = load_json(
        "mythology_universe.json"
    )

    print("Loading characters...")
    characters = load_json(
        "characters.json"
    )

    print("Loading season plan...")
    season_plan = load_json(
        "season_1_plan.json"
    )

    print("Loading episode draft...")
    draft = load_json(
        draft_file
    )

    episode_plan = get_episode_plan(
        season_plan,
        episode_number
    )

    prompt = build_editor_prompt(
        universe,
        characters,
        episode_plan,
        draft
    )

    client = genai.Client(
        api_key=api_key
    )

    last_error = None

    for attempt in range(3):
        try:
            print(
                f"Polishing Episode {episode_number} "
                f"with {MODEL_NAME}..."
            )

            interaction = client.interactions.create(
                model=MODEL_NAME,
                input=prompt
            )

            polished = extract_json(
                interaction.output_text
            )

            polished = validate_final_episode(
                polished,
                episode_number
            )

            output_file = (
                f"episode_{episode_number:02d}_final.json"
            )

            save_json(
                output_file,
                polished
            )

            print("")
            print("Episode polishing successful.")
            print(
                f"Output: {output_file}"
            )
            print(
                "Final calculated runtime: "
                f"{polished['calculated_duration_minutes']} "
                "minutes"
            )

            return

        except Exception as error:
            last_error = error

            print(
                f"Attempt {attempt + 1} failed: {error}"
            )

            if attempt < 2:
                print(
                    "Waiting 15 seconds before retry..."
                )
                time.sleep(15)

    raise RuntimeError(
        "Episode polishing failed after "
        f"3 attempts: {last_error}"
    )


if __name__ == "__main__":
    episode_number = int(
        os.environ.get(
            "EPISODE_NUMBER",
            "1"
        )
    )

    polish_episode(
        episode_number
    )
