import json
import random
import torch
import nlpaug.augmenter.word as naw

# -----------------
# Load user intents
# -----------------
with open("user.json", "r", encoding="utf-8") as f:
    user_intents = json.load(f)

# -----------------
# Slang/paraphrase expansion
# -----------------
slang_variants = {
    "you": ["u", "ya"],
    "your": ["ur"],
    "favorite": ["fav"],
    "do you": ["dya", "do ya"],
    "what is": ["whats", "what's"],
    "are you": ["r u"],
    "hello": ["yo", "sup"],
    "hi": ["hey", "yo"]
}

def add_variants(phrase):
    variants = [phrase]
    words = phrase.split()
    for i, w in enumerate(words):
        lw = w.lower()
        if lw in slang_variants:
            alt = random.choice(slang_variants[lw])
            new = words[:i] + [alt] + words[i+1:]
            variants.append(" ".join(new))
    return list(set(variants))

augmented = {}
for intent, examples in user_intents.items():
    expanded = []
    for ex in examples:
        expanded.extend(add_variants(ex))
    augmented[intent] = expanded

with open("user_augmented.json", "w", encoding="utf-8") as f:
    json.dump(augmented, f, indent=2, ensure_ascii=False)

print("✅ Expanded user.json into user_augmented.json with slang variants")

# -----------------
# Facts (kept separate, not injected into dataset)
# -----------------
facts = {
    "location": "Richmond Hill",
    "school": "Electrical Engineering at Waterloo",
    "badminton_rank": "1st in Canada in doubles (Aug 2025)",
    "games": ["MapleStory", "Genshin Impact", "League of Legends", "Marvel Rivals"],
    "foods": ["noodles with soup", "sushi", "ramen", "pho"],
    "term": "2B term",
    "music": ["upbeat anime OSTs", "lofi beats"],
    "hobbies": ["playing video games", "coding", "badminton"]
}

# -----------------
# Work experience
# -----------------
work_experience = [
    {
        "position": "Engineering Intern",
        "company": "Qualcomm, Markham",
        "duration": "May 2025 - Aug 2025",
        "details": [
            "Built the backend for a web-based internal testing tool enabling ~40 developers to rerun AI test cases for git commit regression testing, cutting manual test setup time by 50%.",
            "Engineered a PostgreSQL database and Python pipeline to parse and store over 10,000 nightly and staged test runs.",
            "Developed FastAPI services and integrated with a Streamlit UI to deliver dynamic filtering by SOC, test type, suite, and run date.",
            "Collaborated cross-functionally to integrate client-side execution, allowing automated bisection and graphical display of past test runs."
        ]
    },
    {
        "position": "Data Analyst",
        "company": "CIBC Mellon, Toronto",
        "duration": "Sept 2024 - Dec 2024",
        "details": [
            "Developed automated tools in Power BI to streamline monthly reporting processes, reducing time by 50%.",
            "Created and optimized data models to resolve mapping errors and circular dependencies.",
            "Performed monthly updates to financial datasets with Power Query and DAX while maintaining historical integrity.",
            "Built dashboards that delivered insights influencing operational and financial decision-making."
        ]
    },
    {
        "position": "IT Support Technician",
        "company": "Einfolab Inc, Richmond Hill",
        "duration": "Jan 2024 - Apr 2024",
        "details": [
            "Provided technical support for hardware and software issues for 200+ client companies.",
            "Performed maintenance on networks and servers to optimize performance.",
            "Contributed to a client network migration project relocating over 100 computers.",
            "Documented recurring issues and resolutions to speed up future troubleshooting."
        ]
    }
]

# -----------------
# Projects
# -----------------
projects = [
    {
        "title": "Multi-Country Time Series Data Management System",
        "duration": "May 2025 - July 2025",
        "details": [
            "Developed a C++ system to manage and analyze multi-country time series datasets.",
            "Implemented dynamic arrays, linked lists, hash maps, and graph-based structures for efficient storage and lookup.",
            "Built CSV parsing, data manipulation, and statistical analysis features to ensure scalability."
        ]
    },
    {
        "title": "Alarm Clock",
        "duration": "Sep 2023 - Nov 2023",
        "details": [
            "Designed and built a functional alarm clock for ECE 198 (Design Studio Course).",
            "Developed and wired an STM32 system with adjustable time and alarm settings.",
            "Connected microcontroller, digital display, passive buzzer, and buttons."
        ]
    },
    {
        "title": "AI Chatbot Design",
        "duration": "Aug 2025 - Present",
        "details": [
            "Designed an AI chatbot capable of understanding user intents and providing dynamic responses.",
            "Implemented natural language understanding, response templates, and intent matching.",
            "Worked on personalized recommendations and adaptive dialogue for natural interactions."
        ]
    }
]

# -----------------
# Context memory
# -----------------
class Context:
    def __init__(self):
        self.last_work = None
        self.last_project = None
        self.awaiting_work_details = False
        self.awaiting_project_details = False
        self.has_discussed_work = False
        self.has_discussed_projects = False

context = Context()

# -----------------
# Static bot responses
# -----------------
bot_responses = {
    "greeting": [
        "Hey! How's it going?",
        "Hello! How are you today?",
        "Hi there! Nice to meet you."
    ],
    "about": [
        "My name is Daniel Leung and I'm a student from Richmond Hill who loves playing video games.",
        "My name is Daniel Leung and I enjoy coding and live in Richmond Hill.",
        "My name is Daniel Leung and I'm currently studying Electrical Engineering at Waterloo."
    ],
    "school": [
        "I'm in my 2B term studying Electrical Engineering at Waterloo.",
        "My current focus is Electrical Engineering in my 2B term.",
        "I'm a 2B student at Waterloo in Electrical Engineering."
    ],
    "games": [
        "I really like MapleStory!",
        "My favorite game is Genshin Impact.",
        "I spend a lot of time playing League of Legends.",
        "Recently, I've been playing Marvel Rivals."
    ],
    "food": [
        "My favorite food is noodles with soup.",
        "I enjoy eating sushi.",
        "I'm a fan of ramen.",
        "Pho is one of my favorite dishes."
    ],
    "sports": [
        "I play competitive badminton and I'm currently 1st in Canada in doubles (Aug 2025).",
        "Badminton is my sport, and I'm ranked 1st in Canada in doubles (Aug 2025).",
        "I like badminton — I'm currently ranked #1 in doubles in Canada."
    ],
    "hobbies": [
        "I usually spend my free time playing video games.",
        "My hobbies include coding.",
        "In my free time, I enjoy badminton."
    ],
    "music": [
        "I enjoy listening to upbeat anime OSTs.",
        "My favorite music includes lofi beats.",
        "I'm into anime OSTs and chill background tracks."
    ],
    "achievements": [
        "I'm currently ranked 1st in Canada in doubles badminton (Aug 2025).",
        "I am the national champion in doubles as of August 2025.",
        "I received the Engineering Entrance Scholarship in 2023.",
        "I was awarded the President's Scholarship of Distinction in 2023."
    ]
}

# -----------------
# Dynamic responses
# -----------------
def get_work_response():
    context.last_work = random.choice(work_experience)
    context.awaiting_work_details = True
    context.has_discussed_work = True
    return f"I worked as a {context.last_work['position']} at {context.last_work['company']}."

def get_project_response():
    context.last_project = random.choice(projects)
    context.awaiting_project_details = True
    context.has_discussed_projects = True
    return f"I worked on {context.last_project['title']}."

def get_work_details_response():
    context.awaiting_work_details = False
    details = random.sample(context.last_work["details"], k=min(2, len(context.last_work["details"])))
    return f"More about my {context.last_work['position']} role: " + " ".join(details)

def get_project_details_response():
    context.awaiting_project_details = False
    details = random.sample(context.last_project["details"], k=min(2, len(context.last_project["details"])))
    return f"More about {context.last_project['title']}: " + " ".join(details)

def get_work_duration_response():
    context.awaiting_work_details = False
    return f"I worked at {context.last_work['company']} for {context.last_work['duration']}."

def get_project_duration_response():
    context.awaiting_project_details = False
    return f"I worked on {context.last_project['title']} for {context.last_project['duration']}."

# -----------------
# Response generator
# -----------------
def generate_bot_response(intent):
    if intent == "work": return get_work_response()
    if intent == "work_details": return get_work_details_response()
    if intent == "work_duration": return get_work_duration_response()
    if intent == "projects": return get_project_response()
    if intent == "project_details": return get_project_details_response()
    if intent == "project_duration": return get_project_duration_response()
    return random.choice(bot_responses.get(intent, ["I don't know about that."]))

# -----------------
# Dialogue generator
# -----------------
with open("user_augmented.json", "r", encoding="utf-8") as f:
    user_intents = json.load(f)

# -----------------
# Dialogue generator with profile markers
# -----------------
def generate_dialogue():
    dialogue = []
    context.__init__()

    # Build profile block
    profile_text = "<|profile|>\n"
    for key, value in facts.items():
        profile_text += f"- {key}: {value}\n"
    profile_text += "<|endprofile|>\n"

    num_exchanges = random.randint(4, 6)

    for _ in range(num_exchanges):
        possible_intents = ["greeting", "about", "school", "games", "food",
                            "sports", "hobbies", "music", "achievements"]

        if not context.has_discussed_work:
            possible_intents.append("work")
        elif context.awaiting_work_details:
            possible_intents = ["work_details", "work_duration"]

        if not context.has_discussed_projects:
            possible_intents.append("projects")
        elif context.awaiting_project_details:
            possible_intents = ["project_details", "project_duration"]

        intent = random.choice(possible_intents)

        user_line = f"{random.choice(user_intents[intent])} [intent: {intent}]"
        bot_line = generate_bot_response(intent)

        dialogue.append(f"User: {user_line}")
        dialogue.append(f"Bot: {bot_line}")

    dialogue.insert(0, profile_text)  # <-- prepend profile block
    dialogue.append("<|endoftext|>")
    return {"dialogue": dialogue}

# -----------------
# Build dataset
# -----------------
num_dialogues = 10000
new_dialogues = [generate_dialogue() for _ in range(num_dialogues)]

try:
    with open("chat_data.json", "r", encoding="utf-8") as f:
        existing_data = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    existing_data = []

all_dialogues = existing_data + new_dialogues

with open("chat_data.json", "w", encoding="utf-8") as f:
    json.dump(all_dialogues, f, indent=2, ensure_ascii=False)

print(f"✅ Added {num_dialogues} new dialogues. Total now: {len(all_dialogues)} in chat_data.json")
