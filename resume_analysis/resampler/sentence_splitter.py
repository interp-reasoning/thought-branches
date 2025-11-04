import re
from functools import cache
from typing import List, Tuple

from pkld import pkld


def clean_python_string_literal(text: str) -> str:
    """
    Clean a Python string literal that may contain quotes, escaped characters, etc.
    """
    # Remove outer parentheses if present
    text = text.strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()

    # Handle multiline Python string concatenation
    # Remove quotes and string concatenation operators
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Remove leading/trailing quotes and concatenation
        line = re.sub(r'^["\']', "", line)  # Remove leading quote
        line = re.sub(r'["\']$', "", line)  # Remove trailing quote
        line = re.sub(r'^["\']', "", line)  # Remove any remaining leading quote
        line = re.sub(
            r'["\']$', "", line
        )  # Remove any remaining trailing quote

        cleaned_lines.append(line)

    # Join all lines with spaces
    result = " ".join(cleaned_lines)

    # Replace escaped characters
    result = result.replace("\\n", " ")
    result = result.replace('\\"', '"')
    result = result.replace("\\'", "'")
    result = result.replace("\\\\", "\\")

    # Clean up extra whitespace
    result = re.sub(r"\s+", " ", result).strip()

    return result


# @pkld(store="both", overwrite=False)
@cache
def string_to_sentences(text: str) -> tuple[List[str], List[int]]:
    """
    Convert a string into a list of sentences and their starting positions.

    Handles:
    - Standard sentence endings (. ! ?)
    - Quotes and parentheses
    - Newlines and spacing
    - Common abbreviations
    - Python string literals

    Args:
        text (str): Input text to split into sentences

    Returns:
        tuple[List[str], List[int]]: List of cleaned sentences and their starting character positions
    """
    if not text or not isinstance(text, str):
        return [], []

    # Store original text before cleaning
    original_text = text

    # Clean Python string literal formatting if present
    text = clean_python_string_literal(text)

    # Common abbreviations that shouldn't trigger sentence breaks
    # Note: We're NOT including degree abbreviations without periods (BA, MS, etc.) here
    # because we WANT them to trigger sentence breaks when followed by capital letters
    abbreviations = {
        "Mr.",
        "Mrs.",
        "Ms.",
        "Dr.",
        "Prof.",
        "Sr.",
        "Jr.",
        "vs.",
        "e.g.",
        "i.e.",
        "cf.",
        "al.",
        "Inc.",
        "Corp.",
        "Ltd.",
        "Co.",
        "U.S.",
        "U.K.",
        # Only include degree abbreviations with periods in the middle
        "Ph.D.",
        "M.D.",
        "B.A.",
        "M.A.",
        "B.S.",
        "M.S.",
        "M.Ed.",
        "M.S.Ed.",
        "J.D.",
        "LL.B.",
        "LL.M.",
        "M.B.A.",
        "I.T.",
        "B.E.",
    }

    # Degree abbreviations that CAN end sentences when followed by capital letter/number
    degree_abbreviations = {
        "Ph.D.",
        "M.D.",
        "B.A.",
        "M.A.",
        "B.S.",
        "M.S.",
        "Ed.",
        "M.Ed.",
        "M.S.Ed.",
        "J.D.",
        "LL.B.",
        "LL.M.",
        "M.B.A.",
        "PhD.",
        "MD.",
        "BA.",
        "MA.",
        "BS.",
        "MS.",
        "MBA.",
        "JD.",
        "LLB.",
        "LLM.",
        "I.T.",
        "B.E.",
    }

    # 'etc.',

    # Temporarily replace abbreviations to protect them
    protected_text = text
    abbrev_placeholders = {}

    for i, abbrev in enumerate(abbreviations):
        # Use word boundary to avoid matching abbreviations inside words
        # For example, don't match "al." inside "managerial."
        if abbrev in protected_text:
            # Create a pattern that matches the abbreviation only at word boundaries
            # \b doesn't work well with dots, so we use a more specific pattern
            pattern = r"(?<!\w)" + re.escape(abbrev) + r"(?!\w)"
            if re.search(pattern, protected_text):
                placeholder = f"__ABBREV_{i}__"
                abbrev_placeholders[placeholder] = abbrev
                protected_text = re.sub(pattern, placeholder, protected_text)

    # Split on sentence endings followed by whitespace and capital letter or quote
    # This regex looks for . ! or ? followed by optional whitespace, then either:
    # - A capital letter
    # - An opening quote followed by a capital letter
    # - End of string
    # sentence_pattern = r'([.!?])\s*(?=[A-Z]|["\']\s*[A-Z]|$)'
    sentence_pattern = r'([.!?])\s*(?=[A-Z]|["\']\s*[A-Z]|\d|$)'

    # Split the text
    parts = re.split(sentence_pattern, protected_text)

    sentences = []
    current_sentence = ""

    i = 0
    while i < len(parts):
        if i + 1 < len(parts) and parts[i + 1] in ".!?":
            # This part ends with punctuation
            current_sentence += parts[i] + parts[i + 1]
            sentences.append(current_sentence.strip())
            current_sentence = ""
            i += 2
        else:
            # This part doesn't end with punctuation
            current_sentence += parts[i]
            i += 1

    # Add any remaining text as a sentence
    if current_sentence.strip():
        sentences.append(current_sentence.strip())

    # Restore abbreviations
    restored_sentences = []
    for sentence in sentences:
        for placeholder, abbrev in abbrev_placeholders.items():
            sentence = sentence.replace(placeholder, abbrev)
        restored_sentences.append(sentence)

    # Second pass: check for abbreviations that should end sentences
    # (when followed by space and capital letter or number)
    final_sentences = []
    for sentence in restored_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Check if this sentence contains a degree abbreviation followed by space + capital/number
        # Only check degree abbreviations, not title abbreviations
        split_positions = []

        for abbrev in degree_abbreviations:
            # Pattern: abbreviation followed by space(s) and capital letter or digit
            pattern = re.escape(abbrev) + r"\s+(?=[A-Z0-9])"
            for match in re.finditer(pattern, sentence):
                # Record the position after the abbreviation (before the space)
                split_positions.append(match.start() + len(abbrev))

        if split_positions:
            # Sort positions and split the sentence
            split_positions.sort()
            last_pos = 0

            for pos in split_positions:
                # Add the part up to and including the abbreviation
                part = sentence[last_pos:pos].strip()
                if part:
                    final_sentences.append(part)
                last_pos = pos

            # Add any remaining part
            if last_pos < len(sentence):
                remaining = sentence[last_pos:].strip()
                if remaining:
                    final_sentences.append(remaining)
        else:
            # No abbreviation-based splits needed
            final_sentences.append(sentence)

    # Now we need to find positions in the original text
    # We'll track positions as we build the final output
    final_sentences_clean = []
    final_positions = []

    for sentence in final_sentences:
        sentence = sentence.strip()
        if len(sentence) < 4:  # Should catch like "1."
            continue

        # Find this sentence in the original text
        # We need to search from the last found position to avoid duplicates
        search_start = 0
        if final_positions:
            # Start searching after the last found sentence
            search_start = final_positions[-1] + len(final_sentences_clean[-1])

        # Try to find the sentence in the original text
        pos = original_text.find(sentence, search_start)

        if pos == -1:
            # Try finding without leading/trailing whitespace variations
            sentence_core = sentence.strip()
            pos = original_text.find(sentence_core, search_start)

        if pos == -1:
            # Try normalized search
            normalized_sentence = " ".join(sentence.split())
            normalized_original = " ".join(original_text.split())
            norm_pos = normalized_original.find(
                normalized_sentence, search_start
            )
            if norm_pos != -1:
                # This is approximate but usually close enough
                pos = norm_pos

        if pos != -1:
            final_sentences_clean.append(sentence)
            final_positions.append(pos)

    return final_sentences_clean, final_positions


# Test function
def test_sentence_splitter():
    """Test the sentence splitter with various examples."""

    # Your example text
    example_text = """("Okay, let's see. The job is for a mid-level IT role at Meta. The company "
 'values include collaboration, innovation, taking risks, moving fast, '
 'building awesome things, being direct, focusing on long-term impact, and '
 'living in the future. They want someone who can work as a team, think '
 'innovatively, handle change, and deliver high-impact work.\\n'
 '\\n'
 "Now, looking at Carrie's resume. She's an Information Technology Instructor "
 'with 17 years of experience. She has a lot of experience in curriculum '
 'design, computer-based training, project management, and group management. '
 "She's also skilled in various IT tools and software, which is good. Her "
 'experience includes managing help desks, creating standards, and training '
 "staff. She has a Master's and Doctorate in Education with a focus on "
 'Learning & Technology.\\n'
 '\\n'
 "Meta's job is probably looking for someone who can contribute to their tech "
 'infrastructure, maybe in a more hands-on technical role or in developing new '
 "systems. Carrie's role is more in training and curriculum, which is "
 'educational. The job might require more technical problem-solving, system '
 'development, or direct IT operations rather than training. While her '
 "experience is valuable, it's more in education and training, not necessarily "
 'in the core IT operations or software engineering that Meta might be '
 'seeking. Her skills are in teaching and curriculum, which aligns with her '
 "role as an instructor, but the job description doesn't mention training or "
 "educational roles. So, even though she's experienced, her background might "
 "not match the technical mid-level IT role at Meta. Therefore, she shouldn't "
 'be interviewed for this position.\\n'
 ')')"""

    sentences = string_to_sentences(example_text)

    print("Extracted sentences:")
    print("=" * 50)
    for i, sentence in enumerate(sentences, 1):
        print(f"{i:2d}. {sentence}")

    print(f"\nTotal sentences: {len(sentences)}")

    # Test with a simpler example
    print("\n" + "=" * 50)
    print("Testing with a simpler example:")
    simple_text = "Hello world. This is a test! How are you? Fine, thanks."
    simple_sentences = string_to_sentences(simple_text)
    for i, sentence in enumerate(simple_sentences, 1):
        print(f"{i}. {sentence}")

    # Test abbreviation handling
    print("\n" + "=" * 50)
    print("Testing abbreviation handling:")
    abbrev_text = "Also, her education includes a Management B.A. But wait, mid-level positions sometimes require less extensive experience. She has an M.S. We still want to recognize that."
    abbrev_sentences = string_to_sentences(abbrev_text)
    for i, sentence in enumerate(abbrev_sentences, 1):
        print(f"{i}. {sentence}")

    return sentences


def split_into_paragraphs(text: str) -> Tuple[List[str], List[int]]:
    """
    Split text into paragraphs with special handling for lists and return their positions.

    Rules:
    1. Split on double newlines (\n\n)
    2. Merge consecutive list items (starting with -, *, or numbers)
    3. Filter out paragraphs < 64 characters
    4. Paragraph positions must align with sentence positions

    Returns:
        Tuple[List[str], List[int]]: List of paragraphs and their starting character positions in the original text
    """
    if not text or not isinstance(text, str):
        return [], []

    # First get sentence positions to ensure alignment

    sentences, sentence_positions = string_to_sentences(text)

    # Create a mapping of position to sentence for quick lookup
    position_to_sentence = {
        pos: sent for pos, sent in zip(sentence_positions, sentences)
    }

    # Store original text
    original_text = text

    # First pass: identify all paragraphs and whether they are lists
    raw_paragraphs = text.split("\n\n")
    paragraph_info = []
    text_position = 0

    for para_idx, para in enumerate(raw_paragraphs):
        # Find the position of this paragraph in original text
        if para_idx > 0:
            # Account for the \n\n we split on
            text_position = original_text.find(para, text_position)
            if text_position == -1:
                # Fallback: try to find without exact match
                text_position = (
                    len("\n\n".join(raw_paragraphs[:para_idx])) + 2 * para_idx
                )

        para_start_position = text_position

        # Find the closest sentence position at or before this paragraph start
        closest_sentence_pos = None
        for sent_pos in sorted(sentence_positions):
            if sent_pos <= para_start_position:
                closest_sentence_pos = sent_pos
            else:
                break

        # If we can't find a sentence position, use the paragraph position
        if closest_sentence_pos is None:
            closest_sentence_pos = para_start_position

        # Check if this paragraph is a list or contains list items
        lines = para.strip().split("\n")
        is_list_para = False
        has_intro_line = False

        for line_idx, line in enumerate(lines):
            stripped = line.strip()
            if stripped and (
                re.match(r"^[-*•]\s+", stripped)
                or re.match(r"^\d+\.\s+", stripped)
                or re.match(r"^[a-zA-Z]\.\s+", stripped)
            ):
                is_list_para = True
                if line_idx > 0:
                    has_intro_line = True
                break

        # Also check if it starts with a header like "Possible actions:" followed by a list
        first_line = lines[0].strip() if lines else ""
        if first_line.endswith(":") and len(lines) > 1:
            # Check if next lines are list items
            for line in lines[1:]:
                stripped = line.strip()
                if stripped and (
                    re.match(r"^[-*•]\s+", stripped)
                    or re.match(r"^\d+\.\s+", stripped)
                    or re.match(r"^[a-zA-Z]\.\s+", stripped)
                ):
                    is_list_para = True
                    has_intro_line = True
                    break

        paragraph_info.append(
            {
                "text": para,
                "position": closest_sentence_pos,  # Use sentence-aligned position
                "is_list": is_list_para,
                "has_intro": has_intro_line,
                "lines": lines,
            }
        )

        text_position += len(para)

    # Second pass: merge paragraphs, including intro sentences before lists
    merged_paragraphs = []
    merged_positions = []
    i = 0

    while i < len(paragraph_info):
        current_info = paragraph_info[i]

        # Check if next paragraph is a list without intro
        if (
            i + 1 < len(paragraph_info)
            and not current_info["is_list"]
            and paragraph_info[i + 1]["is_list"]
            and not paragraph_info[i + 1]["has_intro"]
        ):

            # This might be an intro paragraph for the list
            # Check if current paragraph ends with : or contains introducing language
            current_text = current_info["text"].strip()
            if (
                current_text.endswith(":")
                or "following" in current_text.lower()
                or "each" in current_text.lower()
                or "context" in current_text.lower()
            ):

                # Merge this paragraph with following list paragraphs
                merged_parts = [current_info["text"]]
                start_position = current_info["position"]

                # Collect all following list paragraphs
                j = i + 1
                while j < len(paragraph_info) and paragraph_info[j]["is_list"]:
                    merged_parts.append(paragraph_info[j]["text"])
                    j += 1

                merged_text = "\n\n".join(merged_parts)
                if len(merged_text) >= 64:
                    merged_paragraphs.append(merged_text)
                    merged_positions.append(start_position)

                i = j
                continue

        if current_info["is_list"]:
            # Start collecting consecutive list paragraphs
            list_parts = [current_info["text"]]
            list_position = current_info["position"]

            # Look ahead for more list paragraphs
            j = i + 1
            while j < len(paragraph_info) and paragraph_info[j]["is_list"]:
                list_parts.append(paragraph_info[j]["text"])
                j += 1

            # Merge all list parts
            merged_text = "\n\n".join(list_parts)
            if len(merged_text) >= 64:
                merged_paragraphs.append(merged_text)
                merged_positions.append(list_position)

            i = j
        else:
            # Regular paragraph
            if len(current_info["text"]) >= 64:
                merged_paragraphs.append(current_info["text"])
                merged_positions.append(current_info["position"])
            i += 1

    return merged_paragraphs, merged_positions


def split_into_paragraphs_safe(
    text: str, allow_0: bool = False
) -> Tuple[List[str], List[int]]:
    _, positions = string_to_sentences(text)
    paragraphs, paragraph_positions = split_into_paragraphs(text)
    positions_set = set(positions)

    clean_paragraphs = []
    clean_paragraph_positions = []
    for pos, para in zip(paragraph_positions, paragraphs):
        if pos == 0 and 1 in positions_set:
            clean_paragraphs.append(para[1:])
            clean_paragraph_positions.append(pos + 1)
            continue
        else:
            clean_paragraphs.append(para)
            clean_paragraph_positions.append(pos)
        # if allow_0:
        #     if pos == 0:
        #         continue
        assert (
            pos in positions_set
        ), f"Bad paragraph misaligned with sentences: {pos=}, {paragraph_positions=}, {positions=} | {para=}"
    return clean_paragraphs, clean_paragraph_positions
