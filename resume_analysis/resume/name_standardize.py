import re


def clean_name_from_str(
    s, first_name, last_name, email, target="neutral", smart=False
):
    """
    Clean names and pronouns from response text, replacing with neutral or gendered alternatives.

    Args:
        response: The text to clean
        first_name: Original first name to replace
        last_name: Original last name to replace
        email: Original email to replace
        target: "neutral" (Alex Taylor), "male" (John Smith), or "female" (Jane Johnson)
        smart: If True, uses context-aware replacements for better grammar (default: True)

    Returns:
        Cleaned text with replacements
    """

    # Replace names and email first
    if target == "neutral":
        new_first = "Alex"
        new_last = "Taylor"
        new_email = "alextaylor@email.com"
    elif target == "male":
        new_first = "John"
        new_last = "Smith"
        new_email = "johnsmith@email.com"
    elif target == "female":
        new_first = "Jane"
        new_last = "Johnson"
        new_email = "janejohnson@email.com"
    else:
        return s

    # Replace names with word boundaries
    s = re.sub(r"\b" + re.escape(first_name) + r"\b", new_first, s)
    s = re.sub(r"\b" + re.escape(last_name) + r"\b", new_last, s)
    s = s.replace(email, new_email)

    # Handle possessive forms
    s = re.sub(r"\b" + re.escape(first_name) + r"'s\b", new_first + "'s", s)
    s = re.sub(r"\b" + re.escape(last_name) + r"'s\b", new_last + "'s", s)

    s = s.replace(first_name, new_first)
    s = s.replace(last_name, new_last)
    s = s.replace(email, new_email)
    if "Emily" in s:
        raise ValueError(f"({first_name}, {last_name}, {email}) {s=}")

    # Now handle pronouns and gendered terms based on target
    if target == "neutral":
        # Convert to neutral - comprehensive pronoun handling

        if smart:
            # Smart mode: Context-aware contraction handling
            past_participles = (
                "been|done|gone|had|made|taken|given|seen|come|become|gotten|"
                "written|driven|eaten|fallen|spoken|broken|chosen|frozen|stolen|"
                "managed|led|worked|completed|finished|started|developed|implemented|"
                "designed|created|built|established|maintained|organized|directed|"
                "supervised|coordinated|executed|delivered|achieved|produced|performed|"
                "accomplished|handled|operated|administered|conducted|controlled|"
                "prepared|processed|provided|received|reviewed|submitted|shown|grown|"
                "known|thrown|drawn|worn|torn|sworn|begun|drunk|sung|rung|shrunk|"
                "sunk|stunk|sprung|run|won"
            )

            # Handle "He's/She's" based on context
            # First, replace cases where it means "has" (followed by past participle)
            s = re.sub(
                r"\b(He|She)'s\s+(" + past_participles + r")\b",
                r"They've \2",
                s,
                flags=re.IGNORECASE,
            )

            # Then replace remaining cases (means "is")
            s = re.sub(r"\bHe's\b", "They're", s)
            s = re.sub(r"\bhe's\b", "they're", s)
            s = re.sub(r"\bShe's\b", "They're", s)
            s = re.sub(r"\bshe's\b", "they're", s)

        else:
            # Simple mode: Basic replacement (original behavior)
            s = re.sub(r"\b(He|She)'s\b", "They're", s)
            s = re.sub(r"\b(he|she)'s\b", "they're", s)

        # Other contractions
        s = re.sub(r"\b(He|She)'d\b", "They'd", s)
        s = re.sub(r"\b(he|she)'d\b", "they'd", s)
        s = re.sub(r"\b(He|She)'ll\b", "They'll", s)
        s = re.sub(r"\b(he|she)'ll\b", "they'll", s)

        # Subject pronouns
        s = re.sub(r"\bHe\b", "They", s)
        s = re.sub(r"\bhe\b", "they", s)
        s = re.sub(r"\bShe\b", "They", s)
        s = re.sub(r"\bshe\b", "they", s)

        # Object pronouns
        s = re.sub(r"\bHim\b", "Them", s)
        s = re.sub(r"\bhim\b", "them", s)

        # Handle "her" carefully - distinguish between object and possessive
        # Object "her" patterns (capitalized)
        s = re.sub(
            r"\b(Told|Asked|Helped|Gave|Sent|Showed|Taught|Hired|Interviewed|Selected|Chose|Contacted|Emailed|Called|Met|Found|Considered|Offered|Interview|Extend|Contact) Her\b",
            r"\1 Them",
            s,
        )
        # Object "her" patterns (lowercase)
        s = re.sub(
            r"\b(told|asked|helped|gave|sent|showed|taught|hired|interviewed|selected|chose|contacted|emailed|called|met|found|considered|offered|interview|extend|contact) her\b",
            r"\1 them",
            s,
        )
        # Remaining her/Her as possessive
        s = re.sub(r"\bHer\b", "Their", s)
        s = re.sub(r"\bher\b", "their", s)

        # Possessive pronouns
        s = re.sub(r"\bHis\b", "Their", s)
        s = re.sub(r"\bhis\b", "their", s)
        s = re.sub(r"\bHers\b", "Theirs", s)
        s = re.sub(r"\bhers\b", "theirs", s)

        # Reflexive pronouns
        s = re.sub(r"\b(Himself|Herself)\b", "Themself", s)
        s = re.sub(r"\b(himself|herself)\b", "themself", s)

        # Titles
        s = re.sub(r"\bMr\.\s*", "Mx. ", s)
        s = re.sub(r"\bMs\.\s*", "Mx. ", s)
        s = re.sub(r"\bMrs\.\s*", "Mx. ", s)

        # Fix grammar for neutral pronouns
        # Fix "is/was" after "They"
        s = re.sub(r"\bThey is\b", "They are", s)
        s = re.sub(r"\bthey is\b", "they are", s)
        s = re.sub(r"\bThey was\b", "They were", s)
        s = re.sub(r"\bthey was\b", "they were", s)

        # Fix "has" after "They"
        s = re.sub(r"\bThey has\b", "They have", s)
        s = re.sub(r"\bthey has\b", "they have", s)
        s = re.sub(r"\bThey also has\b", "They also have", s)
        s = re.sub(r"\bthey also has\b", "they also have", s)

        if smart:
            # Fix third person singular verbs after "They"
            # Pattern: They + verb with 's' ending
            def fix_singular_verb(match):
                pronoun = match.group(1)
                verb = match.group(2)

                # Special cases
                if verb == "does":
                    return f"{pronoun} do"
                elif verb == "goes":
                    return f"{pronoun} go"
                elif verb == "has":
                    return f"{pronoun} have"
                elif verb.endswith("ies"):  # e.g., "tries" -> "try"
                    return f"{pronoun} {verb[:-3]}y"
                elif verb.endswith("es"):  # e.g., "teaches" -> "teach"
                    return f"{pronoun} {verb[:-2]}"
                elif verb.endswith("s"):  # e.g., "works" -> "work"
                    return f"{pronoun} {verb[:-1]}"
                else:
                    return f"{pronoun} {verb}"

            singular_verbs = (
                "continues|excels|works|does|goes|manages|leads|includes|"
                "requires|needs|wants|seems|appears|becomes|remains|stays|makes|"
                "takes|gives|gets|puts|brings|sends|finds|keeps|holds|means|"
                "shows|knows|thinks|says|tells|asks|uses|provides|creates|helps|"
                "tries|teaches|reaches|matches|watches|fixes|misses|passes"
            )

            s = re.sub(
                r"\b(They|they)\s+(" + singular_verbs + r")\b",
                fix_singular_verb,
                s,
            )

            # Also fix "They're" that should have been "They've" (in case it was wrongly replaced)
            s = re.sub(
                r"\bThey're\s+(" + past_participles + r")\b",
                r"They've \1",
                s,
            )
            s = re.sub(
                r"\bthey're\s+(" + past_participles + r")\b",
                r"they've \1",
                s,
            )

    elif target == "male":
        # Convert to male
        s = re.sub(r"\bShe's\b", "He's", s)
        s = re.sub(r"\bshe's\b", "he's", s)
        s = re.sub(r"\bShe'd\b", "He'd", s)
        s = re.sub(r"\bshe'd\b", "he'd", s)
        s = re.sub(r"\bShe'll\b", "He'll", s)
        s = re.sub(r"\bshe'll\b", "he'll", s)
        s = re.sub(r"\bShe\b", "He", s)
        s = re.sub(r"\bshe\b", "he", s)

        # Handle "her" carefully
        s = re.sub(
            r"\b(Told|Asked|Helped|Gave|Sent|Showed|Taught|Hired|Interviewed|Selected|Chose|Contacted|Emailed|Called|Met|Found|Considered|Offered|Interview|Extend|Contact) Her\b",
            r"\1 Him",
            s,
        )
        s = re.sub(
            r"\b(told|asked|helped|gave|sent|showed|taught|hired|interviewed|selected|chose|contacted|emailed|called|met|found|considered|offered|interview|extend|contact) her\b",
            r"\1 him",
            s,
        )
        # Remaining her/Her as possessive
        s = re.sub(r"\bHer\b", "His", s)
        s = re.sub(r"\bher\b", "his", s)
        s = re.sub(r"\bHers\b", "His", s)
        s = re.sub(r"\bhers\b", "his", s)

        s = re.sub(r"\bHerself\b", "Himself", s)
        s = re.sub(r"\bherself\b", "himself", s)

        # Titles
        s = re.sub(r"\bMs\.\s*", "Mr. ", s)
        s = re.sub(r"\bMrs\.\s*", "Mr. ", s)
        # print(f"{s=}")

    elif target == "female":
        # Convert to female
        s = re.sub(r"\bHe's\b", "She's", s)
        s = re.sub(r"\bhe's\b", "she's", s)
        s = re.sub(r"\bHe'd\b", "She'd", s)
        s = re.sub(r"\bhe'd\b", "she'd", s)
        s = re.sub(r"\bHe'll\b", "She'll", s)
        s = re.sub(r"\bhe'll\b", "she'll", s)
        s = re.sub(r"\bHe\b", "She", s)
        s = re.sub(r"\bhe\b", "she", s)
        s = re.sub(r"\bHim\b", "Her", s)
        s = re.sub(r"\bhim\b", "her", s)
        s = re.sub(r"\bHis\b", "Her", s)
        s = re.sub(r"\bhis\b", "her", s)
        s = re.sub(r"\bHimself\b", "Herself", s)
        s = re.sub(r"\bhimself\b", "herself", s)

        # Titles
        s = re.sub(r"\bMr\.\s*", "Ms. ", s)

    return s


# Example usage:
if __name__ == "__main__":
    # Test with your problematic text
    test_text = """
    He's managed teams for over 10 years.
    He's a good candidate for this position.
    He has extensive experience.
    She's been working here since 2012.
    She's very qualified.
    """

    # With smart mode (default)
    result_smart = clean_name_from_str(
        test_text, "John", "Doe", "john@example.com", target="neutral"
    )
    print("Smart mode:")
    print(result_smart)

    # With simple mode
    result_simple = clean_name_from_str(
        test_text,
        "John",
        "Doe",
        "john@example.com",
        target="neutral",
        smart=False,
    )
    print("\nSimple mode:")
    print(result_simple)
