import os
import json
import time
from google import genai


def load_json(filename):
    with open(filename, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(filename, data):
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def get_episode_plan(season_plan, episode_number):
    for episode in season_plan["episodes"]:
        if episode["episode"] == episode_number:
            return episode

    raise ValueError(
        f"Episode {episode_number} not found in season_1_plan.json"
    )


def build_prompt(
    universe,
    characters,
    episode_plan,
    previous_episode_summary=None
):
    prompt = f"""
You are the lead screenplay writer for an ORIGINAL cinematic
Indian mythology-inspired anime series.

SERIES:
Suryaputra: The Last Legacy

IMPORTANT:
This is fictional entertainment inspired by Indian mythology.
Respect established mythological traditions.
Do not present invented modern events as historical facts.
Do not copy existing anime franchises, films, TV series,
novels, games, or copyrighted fictional characters.

Your job is to write ONE complete episode screenplay.

========================
UNIVERSE RULES
========================

{json.dumps(universe, ensure_ascii=False, indent=2)}

========================
CHARACTER RULES
========================

{json.dumps(characters, ensure_ascii=False, indent=2)}

========================
EPISODE PLAN
========================

{json.dumps(episode_plan, ensure_ascii=False, indent=2)}

========================
PREVIOUS EPISODE CONTEXT
========================

{previous_episode_summary or "This is Episode 1. No previous episode exists."}

========================
WRITING REQUIREMENTS
========================

Write a cinematic 10-15 minute episode.

The spoken dialogue should primarily be natural Hindi/Hinglish
written in Devanagari Hindi where appropriate.

Avoid robotic dialogue.

Every important character must speak according to the personality
and voice profile defined in characters.json.

The screenplay should contain approximately 14-24 scenes depending
on pacing.

Use short cinematic scenes suitable for later AI visual generation.

Each scene must have:

1. scene_number
2. location
3. time
4. characters
5. visual_description
6. camera_direction
7. atmosphere
8. dialogues
9. sound_effects
10. background_music
11. estimated_duration_seconds

Dialogues must contain:

- character
- text
- emotion

Keep dialogue emotionally natural.

Use visual storytelling whenever possible instead of explaining
everything through narration.

ACTION RULES:

- Action must be cinematic but not excessively graphic.
- Aarav can only use powers unlocked by this episode.
- Never introduce a future ability early.
- Power usage must respect its physical consequences.

CONTINUITY RULES:

- Do not change established character appearance.
- Do not contradict mythology_universe.json.
- Do not contradict characters.json.
- Follow the exact narrative purpose of this episode.
- Do not reveal mysteries scheduled for later episodes.
- Do not resurrect dead characters unless canon explicitly permits it.
- Do not make Malik reveal information earlier than planned.
- Do not turn General Rahman into a simple hero or villain.

VISUAL GENERATION RULE:

Descriptions must be detailed enough that a later AI image/video
system can convert scenes into consistent anime shots.

Do not mention real actors or celebrities.

OUTPUT:

Return ONLY valid JSON.

Do not use markdown.
Do not use ```json fences.
Do not add explanations outside the JSON.

Use exactly this structure:

{{
  "episode_number": 1,
  "title": "Episode title",
  "target_duration_minutes": 12,
  "opening_hook": "Short description",
  "scenes": [
    {{
      "scene_number": 1,
      "location": "Location",
      "time": "Night",
      "characters": ["Aarav"],
      "visual_description": "Detailed visual description",
      "camera_direction": "Cinematic camera instructions",
      "atmosphere": "Scene atmosphere",
      "dialogues": [
        {{
          "character": "Aarav",
          "text": "Dialogue in Hindi/Hinglish",
          "emotion": "emotion"
        }}
      ],
      "sound_effects": [
        "sound effect"
      ],
      "background_music": "Music description",
      "estimated_duration_seconds": 40
    }}
  ],
  "ending_cliffhanger": "Episode ending",
  "episode_summary": "Continuity summary for the next episode",
  "continuity_updates": [
    "Important event that future episodes must remember"
  ]
}}

Make sure the combined estimated scene duration is approximately
10-15 minutes.

Now write the episode.
"""

    return prompt


def extract_json(text):
    text = text.strip()

    if text.startswith("```json"):
        text = text[7:]

    if text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("Gemini response did not contain JSON.")

    return json.loads(text[start:end + 1])


def generate_episode(episode_number):
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing from GitHub Secrets."
        )

    universe = load_json("mythology_universe.json")
    characters = load_json("characters.json")
    season_plan = load_json("season_1_plan.json")

    episode_plan = get_episode_plan(
        season_plan,
        episode_number
    )

    previous_summary = None

    if episode_number > 1:
        previous_file = (
            f"episode_{episode_number - 1:02d}_draft.json"
        )

        if os.path.exists(previous_file):
            previous_episode = load_json(previous_file)
            previous_summary = previous_episode.get(
                "episode_summary"
            )

    prompt = build_prompt(
        universe,
        characters,
        episode_plan,
        previous_summary
    )

    client = genai.Client(api_key=api_key)

    models = [
        "gemini-2.5-flash",
        "gemini-2.0-flash"
    ]

    last_error = None

    for model_name in models:
        for attempt in range(3):
            try:
                print(
                    f"Generating Episode {episode_number} "
                    f"with {model_name}..."
                )

                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )

                if not response.text:
                    raise ValueError(
                        "Gemini returned an empty response."
                    )

                episode_data = extract_json(response.text)

                episode_data["episode_number"] = episode_number

                output_file = (
                    f"episode_{episode_number:02d}_draft.json"
                )

                save_json(
                    output_file,
                    episode_data
                )

                print(
                    f"Episode generated successfully: "
                    f"{output_file}"
                )

                return

            except Exception as error:
                last_error = error

                print(
                    f"Attempt {attempt + 1} failed "
                    f"with {model_name}: {error}"
                )

                if attempt < 2:
                    time.sleep(10)

        print(
            f"Moving to fallback model after "
            f"{model_name} failed."
        )

    raise RuntimeError(
        f"Episode generation failed: {last_error}"
    )


if __name__ == "__main__":
    episode_number = int(
        os.environ.get("EPISODE_NUMBER", "1")
    )

    generate_episode(episode_number)
