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


def get_episode_plan(season_plan, episode_number):
    for episode in season_plan["episodes"]:
        if episode["episode"] == episode_number:
            return episode

    raise ValueError(
        f"Episode {episode_number} not found in season_1_plan.json"
    )


def get_previous_context(episode_number):
    if episode_number <= 1:
        return "This is Episode 1. No previous episode exists."

    previous_file = f"episode_{episode_number - 1:02d}_draft.json"

    if not os.path.exists(previous_file):
        return (
            f"Episode {episode_number - 1} draft is not available. "
            "Follow the season plan carefully and do not invent "
            "major continuity changes."
        )

    previous_episode = load_json(previous_file)

    context = {
        "episode_summary": previous_episode.get("episode_summary", ""),
        "continuity_updates": previous_episode.get(
            "continuity_updates", []
        ),
        "ending_cliffhanger": previous_episode.get(
            "ending_cliffhanger", ""
        )
    }

    return json.dumps(
        context,
        ensure_ascii=False,
        indent=2
    )


def build_prompt(
    universe,
    characters,
    season_plan,
    episode_plan,
    previous_context
):
    return f"""
You are the lead screenplay writer for an ORIGINAL cinematic
Indian mythology-inspired anime series.

SERIES:
Suryaputra: The Last Legacy

Your task is to create ONE complete production-ready episode draft.

This is original fictional entertainment inspired by Indian mythology.

Respect established Indian mythological traditions.

Clearly treat invented modern organizations, artifacts, events,
powers and characters as fictional extensions of this series.

Never copy characters, scenes, dialogue, visual designs or plots
from existing anime, movies, television series, novels or games.

==================================================
MASTER UNIVERSE
==================================================

{json.dumps(universe, ensure_ascii=False, indent=2)}

==================================================
CHARACTER DATABASE
==================================================

{json.dumps(characters, ensure_ascii=False, indent=2)}

==================================================
SEASON INFORMATION
==================================================

Season:
{season_plan.get("season", 1)}

Season core:
{json.dumps(season_plan.get("season_core", {}), ensure_ascii=False, indent=2)}

==================================================
CURRENT EPISODE PLAN
==================================================

{json.dumps(episode_plan, ensure_ascii=False, indent=2)}

==================================================
PREVIOUS EPISODE CONTEXT
==================================================

{previous_context}

==================================================
SCREENPLAY REQUIREMENTS
==================================================

Target runtime:
10 to 15 minutes.

Aim for approximately 12 minutes.

Create approximately 14 to 22 cinematic scenes.

The total of all estimated_duration_seconds values should normally
fall between 600 and 900 seconds.

Dialogue should primarily be natural Hindi/Hinglish.

Write Hindi dialogue mainly in Devanagari script.

Use English words naturally where Indian speakers would normally
use them.

Avoid robotic dialogue and unnecessary exposition.

Characters must speak according to their personality and voice
profile from characters.json.

Use visual storytelling whenever possible.

Every scene must be useful to at least one of these:

- plot progression
- character development
- mystery
- emotion
- tension
- action
- foreshadowing

Do not create filler scenes.

==================================================
EPISODE 1 SPECIAL CONTINUITY
==================================================

If this is Episode 1:

Aarav begins with Vasu, Yashoda and Jaya alive.

Their deaths must happen only according to the Episode 1 plan.

Do not treat Aarav as already understanding his supernatural power.

His first golden shield manifestation must feel shocking,
uncontrolled and physically exhausting.

==================================================
POWER RULES
==================================================

Aarav may ONLY use abilities available at this point in the season.

Never introduce abilities scheduled for later episodes.

Divine powers must have consequences and limitations.

Aarav must not become unbeatable.

Early golden energy should primarily behave defensively.

==================================================
MYSTERY RULES
==================================================

Do not reveal future twists early.

Do not reveal Malik's complete identity.

Do not reveal Malik's full objective.

Do not explain every mystery immediately.

General Rahman must remain morally ambiguous.

Foreshadowing is allowed, but it must not spoil later episodes.

==================================================
VISUAL PRODUCTION RULES
==================================================

This screenplay will later be converted into AI-generated
anime shots.

Therefore every visual_description should clearly describe:

- environment
- character positioning
- important movement
- lighting
- weather when relevant
- emotional visual details

Do NOT redesign characters.

Do NOT mention real actors or celebrities.

Do NOT request the exact style of an existing anime franchise.

Use the established original cinematic Indian
mythology-inspired anime aesthetic.

Camera directions should be practical, such as:

- wide establishing shot
- medium shot
- close-up
- over-the-shoulder
- tracking shot
- low-angle shot
- slow push-in
- aerial establishing shot

Avoid impossible or meaningless camera instructions.

==================================================
AUDIO RULES
==================================================

Each spoken line must specify:

- character
- text
- emotion

Sound effects should be specific enough for later production.

Background music descriptions should describe mood and instruments
without referencing copyrighted songs.

==================================================
OUTPUT RULES
==================================================

Return ONLY valid JSON.

Do NOT use Markdown.

Do NOT use ```json.

Do NOT write explanations before or after the JSON.

Use exactly this overall structure:

{{
  "episode_number": {episode_plan["episode"]},
  "title": "{episode_plan["title"]}",
  "target_duration_minutes": 12,
  "opening_hook": "Short description of the opening hook",

  "scenes": [
    {{
      "scene_number": 1,
      "location": "Specific location",
      "time": "Night",

      "characters": [
        "Aarav"
      ],

      "visual_description": "Detailed anime visual description",

      "camera_direction": "Cinematic camera direction",

      "atmosphere": "Mood and environment",

      "dialogues": [
        {{
          "character": "Aarav",
          "text": "Natural Hindi/Hinglish dialogue",
          "emotion": "Emotion"
        }}
      ],

      "sound_effects": [
        "Specific sound effect"
      ],

      "background_music": "Original background score description",

      "estimated_duration_seconds": 40
    }}
  ],

  "ending_cliffhanger": "Detailed episode-ending cliffhanger",

  "episode_summary": "Concise continuity summary for the next episode",

  "continuity_updates": [
    "Important event future episodes must remember"
  ]
}}

Before producing the final JSON, internally verify:

1. The episode follows its season plan.
2. Character identities remain consistent.
3. No future powers were introduced.
4. No future mystery was accidentally revealed.
5. Dialogue sounds natural.
6. Scene durations total roughly 10-15 minutes.
7. The output is valid JSON.

Now generate the complete episode.
"""


def extract_json(text):
    if not text:
        raise ValueError("Gemini returned an empty response.")

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
            "Gemini response did not contain a valid JSON object."
        )

    json_text = text[start:end + 1]

    return json.loads(json_text)


def validate_episode(episode_data, episode_number):
    required_keys = [
        "episode_number",
        "title",
        "scenes",
        "ending_cliffhanger",
        "episode_summary",
        "continuity_updates"
    ]

    for key in required_keys:
        if key not in episode_data:
            raise ValueError(
                f"Generated episode is missing required key: {key}"
            )

    if not isinstance(episode_data["scenes"], list):
        raise ValueError("scenes must be a list.")

    if len(episode_data["scenes"]) == 0:
        raise ValueError("Generated episode contains no scenes.")

    episode_data["episode_number"] = episode_number

    total_duration = 0

    for scene in episode_data["scenes"]:
        duration = scene.get("estimated_duration_seconds", 0)

        if isinstance(duration, (int, float)):
            total_duration += duration

    episode_data["calculated_duration_seconds"] = total_duration
    episode_data["calculated_duration_minutes"] = round(
        total_duration / 60,
        2
    )

    return episode_data


def generate_episode(episode_number):
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing from GitHub Secrets."
        )

    print("Loading mythology universe...")
    universe = load_json("mythology_universe.json")

    print("Loading character database...")
    characters = load_json("characters.json")

    print("Loading Season 1 plan...")
    season_plan = load_json("season_1_plan.json")

    episode_plan = get_episode_plan(
        season_plan,
        episode_number
    )

    previous_context = get_previous_context(
        episode_number
    )

    prompt = build_prompt(
        universe,
        characters,
        season_plan,
        episode_plan,
        previous_context
    )

    client = genai.Client(api_key=api_key)

    last_error = None

    for attempt in range(3):
        try:
            print(
                f"Generating Episode {episode_number} "
                f"with {MODEL_NAME}..."
            )

            interaction = client.interactions.create(
                model=MODEL_NAME,
                input=prompt
            )

            response_text = interaction.output_text

            episode_data = extract_json(
                response_text
            )

            episode_data = validate_episode(
                episode_data,
                episode_number
            )

            output_file = (
                f"episode_{episode_number:02d}_draft.json"
            )

            save_json(
                output_file,
                episode_data
            )

            print("")
            print("Episode generated successfully.")
            print(f"Output: {output_file}")
            print(
                "Calculated runtime: "
                f"{episode_data['calculated_duration_minutes']} minutes"
            )

            return

        except Exception as error:
            last_error = error

            print("")
            print(
                f"Attempt {attempt + 1} failed: {error}"
            )

            if attempt < 2:
                print("Waiting 15 seconds before retry...")
                time.sleep(15)

    raise RuntimeError(
        f"Episode generation failed after 3 attempts: {last_error}"
    )


if __name__ == "__main__":
    episode_number = int(
        os.environ.get(
            "EPISODE_NUMBER",
            "1"
        )
    )

    generate_episode(
        episode_number
    )
