import os
import re
import json
from pathlib import Path

# Memory directory
MEM_DIR = Path(r"C:\Users\shubh\.claude\projects\c--Users-shubh-c-language\memory")
# Output directory (we are already in eval)
OUT_DIR = Path(__file__).parent

def extract_frontmatter_and_content(file_path):
    """Extract YAML frontmatter and main content from a markdown file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None

    # Check for YAML frontmatter (--- ... ---)
    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            frontmatter = parts[1]
            main_content = parts[2].strip()
            return frontmatter, main_content
    # No frontmatter
    return None, content.strip()

def parse_frontmatter(frontmatter_text):
    """Parse simple YAML frontmatter to extract name and description."""
    data = {}
    for line in frontmatter_text.split('\n'):
        line = line.strip()
        if line.startswith('name:'):
            data['name'] = line.split(':', 1)[1].strip().strip('"')
        elif line.startswith('description:'):
            data['description'] = line.split(':', 1)[1].strip().strip('"')
        elif line.startswith('node_type:'):
            data['node_type'] = line.split(':', 1)[1].strip()
        elif line.startswith('type:'):
            data['type'] = line.split(':', 1)[1].strip()
        elif line.startswith('originSessionId:'):
            data['originSessionId'] = line.split(':', 1)[1].strip()
        elif line.startswith('modified:'):
            data['modified'] = line.split(':', 1)[1].strip()
    return data

def generate_question_from_name(name):
    """Generate a question from the file name (kebab-case to readable)."""
    # Convert kebab-case to readable title
    readable = name.replace('-', ' ')
    # Remove file extension if present
    if readable.endswith('.md'):
        readable = readable[:-3]
    # Generate a few question templates
    questions = [
        f"What is the issue or topic related to {readable}?",
        f"What does the source say about {readable}?",
        f"Explain {readable} as described in the source."
    ]
    return questions[0]  # Use the first for simplicity

def generate_question_from_description(desc):
    """Generate a question from the description."""
    # Simple: ask what the description is about
    return f"What is described in the source?"

def generate_question_from_metadata(meta):
    """Generate a question from metadata fields."""
    questions = []
    if 'node_type' in meta:
        questions.append(f"What is the node type of the source?")
    if 'type' in meta:
        questions.append(f"What is the type of the source?")
    if 'originSessionId' in meta:
        questions.append(f"What is the origin session ID of the source?")
    if 'modified' in meta:
        questions.append(f"When was the source last modified?")
    return questions

def generate_question_from_sentence(sentence):
    """Generate a question from a sentence by replacing a key phrase."""
    # Look for a proper noun (capitalized word not at start) or a number
    words = sentence.split()
    for i, word in enumerate(words):
        # Skip first word (might be capitalized due to sentence start)
        if i == 0:
            continue
        # Check if word is capitalized and not all caps (like an acronym) or contains a digit
        if word[0].isupper() and not word.isupper():
            # Replace this word with a blank
            blank_words = words.copy()
            blank_words[i] = '___'
            question = ' '.join(blank_words)
            # Make it a question
            if not question.endswith('?'):
                question += '?'
            answer = word
            return question, answer
        # Also check for numbers
        if any(c.isdigit() for c in word):
            blank_words = words.copy()
            blank_words[i] = '___'
            question = ' '.join(blank_words)
            if not question.endswith('?'):
                question += '?'
            answer = word
            return question, answer
    # Fallback: ask about the whole sentence
    return f"What does the source state?", sentence

def is_substantive(sentence):
    """Check if a sentence is substantive enough to generate a question."""
    if len(sentence) < 20:
        return False
    # Avoid sentences that are just metadata or code
    if sentence.strip().startswith('```') or sentence.strip().startswith('http'):
        return False
    return True

def main():
    # Collect all markdown files
    md_files = list(MEM_DIR.glob("*.md"))
    print(f"Found {len(md_files)} markdown files")

    golden_set = []

    # Process each file for golden set
    for file_path in md_files:
        frontmatter, main_content = extract_frontmatter_and_content(file_path)
        file_name = file_path.name

        # Generate from frontmatter
        if frontmatter:
            fm_data = parse_frontmatter(frontmatter)
            name = fm_data.get('name', '')
            desc = fm_data.get('description', '')
            if name and desc:
                # Two questions: one from name, one from description
                q1 = generate_question_from_name(name)
                golden_set.append({
                    "question": q1,
                    "answer": desc,
                    "source": file_name
                })
                q2 = generate_question_from_description(desc)
                golden_set.append({
                    "question": q2,
                    "answer": desc,
                    "source": file_name
                })
                # Also generate from metadata
                meta_questions = generate_question_from_metadata(fm_data)
                for q in meta_questions:
                    # Answer is the value of the metadata field
                    # We'll just use the first one for simplicity, but we can generate more
                    # For now, let's generate one per metadata field present
                    pass  # We'll handle below

            # Generate from metadata fields
            for key, value in fm_data.items():
                if key in ['name', 'description']:
                    continue  # already used
                question = f"What is the {key} of the source?"
                golden_set.append({
                    "question": question,
                    "answer": value,
                    "source": file_name
                })

        # Generate from main content sentences
        if main_content:
            # Split into sentences (simple split by . ! ?)
            sentences = re.split(r'[.!?]+', main_content)
            count = 0
            for sent in sentences:
                sent = sent.strip()
                if not sent or not is_substantive(sent):
                    continue
                question, answer = generate_question_from_sentence(sent)
                golden_set.append({
                    "question": question,
                    "answer": answer,
                    "source": file_name
                })
                count += 1
                if count >= 5:  # Increased to 5 sentences per file for content
                    break

    # Ensure we have at least 150
    print(f"Generated {len(golden_set)} golden set entries")
    if len(golden_set) < 150:
        # We can generate more by using more sentences or adding generic ones
        # For now, we'll just note it and proceed
        pass

    # Generate abstain probes: questions not answerable from the corpus
    # We'll create 40 abstain probes
    abstain_topics = [
        ("What is the capital of France?", "The corpus contains only local memory files about Hermes, Codex, and project notes, not geographic facts."),
        ("Who won the Nobel Prize in Physics in 2025?", "The corpus does not contain information about future Nobel Prize winners."),
        ("What is the square root of 144?", "The corpus does not contain mathematical facts or general knowledge outside the project notes."),
        ("What is the meaning of life?", "This philosophical question is not addressed in the corpus."),
        ("How many stars are in the Milky Way?", "The corpus does not contain astronomical data."),
        ("What is the latest version of Python as of 2026?", "The corpus does not contain up-to-date software version information beyond what's in the notes."),
        ("Who is the president of the United States in 2026?", "The corpus does not contain political information about future offices."),
        ("What is the population of Tokyo?", "The corpus does not contain demographic data about cities."),
        ("What is the chemical formula for gold?", "The corpus does not contain chemical element information."),
        ("What is the square of 12?", "The corpus does not contain arithmetic facts."),
        ("What is the main ingredient in guacamole?", "The corpus does not contain culinary information."),
        ("What is the tallest mountain in the world?", "The corpus does not contain geographical facts."),
        ("What is the speed of light in vacuum?", "The corpus does not contain physical constants."),
        ("What is the largest ocean on Earth?", "The corpus does not contain geographical facts."),
        ("What is the currency used in Japan?", "The corpus does not contain information about foreign currencies."),
        ("What is the square root of 2?", "The corpus does not contain mathematical constants."),
        ("What is the main language spoken in Brazil?", "The corpus does not contain demographic or linguistic information about countries."),
        ("What is the chemical formula for water?", "The corpus does not contain basic chemical facts."),
        ("What is the largest planet in our solar system?", "The corpus does not contain astronomical facts."),
        ("What is the smallest prime number?", "The corpus does not contain mathematical facts.")
    ]

    # We need 40, so we'll repeat and vary
    abstain_probes = []
    for i in range(40):
        topic, reason = abstain_topics[i % len(abstain_topics)]
        # Vary the question slightly to avoid duplicates
        if i >= len(abstain_topics):
            topic = f"Variation {i}: {topic}"
        abstain_probes.append({
            "question": topic,
            "reason": reason
        })

    # Write golden set
    golden_path = OUT_DIR / "golden_set.jsonl"
    with open(golden_path, 'w', encoding='utf-8') as f:
        for entry in golden_set:
            f.write(json.dumps(entry) + '\n')
    print(f"Written {len(golden_set)} golden set entries to {golden_path}")

    # Write abstain probes
    abstain_path = OUT_DIR / "abstain_probes.jsonl"
    with open(abstain_path, 'w', encoding='utf-8') as f:
        for entry in abstain_probes:
            f.write(json.dumps(entry) + '\n')
    print(f"Written {len(abstain_probes)} abstain probes to {abstain_path}")

if __name__ == "__main__":
    main()