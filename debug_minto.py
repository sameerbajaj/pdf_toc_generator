
import re

# Mocking the necessary parts of the script for testing

def clean_ocr_page_num(text):
    """Clean OCR'd page number text."""
    # Remove spaces
    text = text.replace(" ", "")
    
    # Common OCR substitutions
    replacements = {
        'I': '1', 'l': '1', 'i': '1', '|': '1',
        'O': '0', 'o': '0',
        'S': '5', 's': '5',
        'B': '8',
        'Z': '2'
    }
    
    # If it's already digits, return int
    if text.isdigit():
        return int(text)
        
    # Avoid converting common words that might match the pattern
    if text.upper() in {'SO', 'IS', 'OS', 'OO', 'LL', 'SS', 'TO', 'BY', 'OF', 'IN', 'ON', 'AT', 'AS'}:
        return None

    # Apply replacements
    cleaned = ""
    for char in text:
        if char.isdigit():
            cleaned += char
        elif char in replacements:
            cleaned += replacements[char]
        else:
            # If we encounter an unknown char, it's probably not a number
            return None
            
    if cleaned and cleaned.isdigit():
        # Safety check: if the original text was long (e.g. "Business") and we converted it to "855",
        # it's probably wrong. Only allow replacements for short strings or if mostly digits.
        if len(text) > 4 and sum(c.isdigit() for c in text) == 0:
             return None
             
        return int(cleaned)
    return None

def test_extraction():
    # Simulate lines from the Minto PDF TOC
    # Based on the screenshot
    
    # y-coordinates are estimated. Assuming line height ~12-15.
    mock_lines = [
        {"text": "PART", "x0": 100, "bbox": (100, 50, 150, 62)},
        {"text": "1", "x0": 100, "bbox": (100, 65, 110, 77)},
        {"text": "LOGIC", "x0": 100, "bbox": (100, 80, 150, 92)},
        {"text": "IN", "x0": 100, "bbox": (100, 95, 120, 107)},
        {"text": "WRITING", "x0": 100, "bbox": (100, 110, 160, 122)},
        
        # Gap here
        {"text": "PREFACE", "x0": 50, "bbox": (50, 160, 100, 172)},
        {"text": "INTRODUCTION TO PART ONE", "x0": 50, "bbox": (50, 180, 200, 192)},
        {"text": "The Minto Pyramid Principle: Logic in Writing", "x0": 50, "bbox": (50, 200, 300, 212)},
        
        # The problematic line. Small gap from previous line (200->220 is 20 units)
        {"text": "1 WHY A PYRAMID STRUCTURE? 1", "x0": 50, "bbox": (50, 220, 400, 232)},
        
        {"text": "Sorting into Pyramids 2", "x0": 70, "bbox": (70, 240, 300, 252)},
        {"text": "The Magical Number Seven 3", "x0": 70, "bbox": (70, 260, 300, 272)},
    ]
    
    page_count = 100
    toc_entries = []
    current_title_lines = []
    current_x0 = None
    last_y_bottom = None
    
    print("Processing lines...")
    for i, line in enumerate(mock_lines):
        text = line["text"]
        x0 = line["x0"]
        bbox = line.get("bbox")
        segments = line.get("segments", []) # Mock doesn't have segments, but logic handles it
        
        print(f"Line {i}: '{text}'")
        
        # Check for page number
        page_num = None
        title_part = None
        
        # 1. Standalone page number
        cleaned = clean_ocr_page_num(text)
        if cleaned is not None:
            if cleaned <= page_count + 50:
                page_num = cleaned
        else:
            # 2. Trailing page number
            # (Skipping segment logic for this mock as we provided full text)
            
            # Fallback to regex
            match = re.search(r'^(.*)\s+([0-9IlOoSBZ]{1,4})$', text)
            if match:
                pot_num = clean_ocr_page_num(match.group(2))
                if pot_num is not None and pot_num <= page_count + 50:
                    suffix = match.group(2)
                    prefix = match.group(1)
                    is_valid_num = True
                    if not suffix.isdigit() and len(suffix) >= 2:
                            if suffix.lower() in {'is', 'so', 'to', 'by', 'of', 'in', 'on', 'at', 'as'}:
                                is_valid_num = False
                    
                    if is_valid_num:
                        page_num = pot_num
                        title_part = prefix.strip()
            
            # 3. Just title
            if page_num is None:
                title_part = text
        
        # Logic to handle the parts
        if title_part:
            # Clean title part
            title_part = re.sub(r'\b([A-Z])\s+(?=[A-Z]\b)', r'\1', title_part)
            title_part = re.sub(r'^\d+\.?\s*', '', title_part)
            
            is_valid_part = True
            if re.match(r'^[—_\-\s]+$|^[a-z]{1,3}\s+[A-Z][a-z]\s+', title_part):
                is_valid_part = False
            
            # Stricter checks for new entries
            if not current_title_lines and is_valid_part:
                word_count = len(title_part.split())
                if word_count == 1:
                    common_words = {'PART', 'LOGIC', 'WRITING', 'CHAPTER', 'SECTION', '|'}
                    if not (title_part[0].isupper() and len(title_part) >= 3 and title_part not in common_words):
                        is_valid_part = False
            
            if is_valid_part:
                # Check vertical gap
                if bbox and last_y_bottom is not None:
                    this_y_top = bbox[1]
                    gap = this_y_top - last_y_bottom
                    print(f"  Gap: {gap}")
                    if gap > 20:
                        print("  -> Gap > 20, resetting!")
                        current_title_lines = []
                        current_x0 = None
                
                # NEW LOGIC: Check if this line starts with a number (indicating a new section)
                # while we already have accumulated text that doesn't look like a prefix.
                # We check the 'text' variable or 'title_part' before stripping.
                # But 'title_part' here is already stripped.
                # So we need to check 'text' or capture the unstripped version.
                
                # Let's look at 'text'. It might have the page number at the end.
                # "1 WHY A PYRAMID STRUCTURE? 1"
                # We want to check if it starts with "1 ".
                
                # Simple check: does 'text' start with a digit?
                if current_title_lines and re.match(r'^\d', text.strip()):
                    # Check if previous lines look like a prefix
                    prev_text = " ".join(current_title_lines).upper()
                    is_prefix = False
                    for prefix in ['CHAPTER', 'PART', 'SECTION', 'UNIT', 'MODULE']:
                        if prev_text.endswith(prefix) or prev_text == prefix:
                            is_prefix = True
                            break
                    
                    if not is_prefix:
                        print("  -> New numbered section detected, resetting previous lines!")
                        current_title_lines = []
                        current_x0 = None

                if not current_title_lines:
                    current_x0 = x0
                current_title_lines.append(title_part)
                
                if bbox:
                    last_y_bottom = bbox[3]
            
        if page_num is not None:
            print(f"  -> Found page num: {page_num}")
            if current_title_lines:
                full_title = " ".join(current_title_lines)
                print(f"  -> Committing entry: '{full_title}' -> {page_num}")
                toc_entries.append({
                    "title": full_title,
                    "page": page_num
                })
                current_title_lines = []
                current_x0 = None
                last_y_bottom = None
            else:
                print("  -> No title lines to commit!")

    return toc_entries

if __name__ == "__main__":
    entries = test_extraction()
    print("\nFinal Entries:")
    for e in entries:
        print(f"{e['page']}: {e['title']}")
