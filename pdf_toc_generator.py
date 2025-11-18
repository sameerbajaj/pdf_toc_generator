#!/usr/bin/env python3
"""
PDF Table of Contents Generator
Automatically generates and adds bookmarks to PDF files
"""

import fitz  # PyMuPDF
from collections import Counter
import re
from pathlib import Path
import json
import sys
import argparse


def detect_toc_page(pdf_path, max_pages_to_check=10):
    """Detect if PDF has an existing Table of Contents page(s)."""
    doc = fitz.open(pdf_path)
    toc_patterns = [
        r'\btable\s+of\s+contents\b',
        r'\bcontents\b',
        r'\btoc\b'
    ]
    
    toc_pages = []
    toc_started = False
    
    for page_num in range(min(max_pages_to_check, len(doc))):
        page = doc[page_num]
        text = page.get_text().lower()
        
        for pattern in toc_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                toc_started = True
                toc_pages.append(page_num)
                break
        
        if toc_started and page_num not in toc_pages:
            has_chapters = len(re.findall(r'\bchapter\s+\d+', text, re.IGNORECASE)) > 0
            page_refs = len(re.findall(r'\bpage\s+\d+|\.\.\.\s*\d+|\s+\d+\s*$', text, re.MULTILINE | re.IGNORECASE))
            
            if has_chapters or page_refs > 3:
                toc_pages.append(page_num)
            else:
                break
    
    doc.close()
    
    if toc_pages:
        if len(toc_pages) == 1:
            print(f"✓ Found TOC on page {toc_pages[0] + 1}")
        else:
            print(f"✓ Found TOC spanning pages {toc_pages[0] + 1}-{toc_pages[-1] + 1}")
        return toc_pages
    
    print("ℹ No existing TOC page detected")
    return None


def extract_toc_from_page(pdf_path, toc_page_nums, use_ocr_mode=False):
    """Extract TOC entries from dedicated TOC page(s)."""
    # If OCR mode is enabled, use OCR-style extraction directly
    if use_ocr_mode:
        return extract_toc_ocr_style(pdf_path, toc_page_nums)
    
    # Otherwise, try standard extraction first
    doc = fitz.open(pdf_path)
    
    if isinstance(toc_page_nums, int):
        toc_page_nums = [toc_page_nums]
    
    toc_entries = []
    pending_header = None
    
    # Try standard extraction first
    for toc_page_num in toc_page_nums:
        page = doc[toc_page_num]
        blocks = page.get_text("dict", flags=11)["blocks"]
        
        for block in blocks:
            if block["type"] == 0:
                for line in block["lines"]:
                    line_text = ""
                    line_x = float('inf')
                    font_size = 0
                    
                    for span in line["spans"]:
                        line_text += span["text"]
                        line_x = min(line_x, span["bbox"][0])
                        font_size = max(font_size, span["size"])
                    
                    line_text = line_text.strip()
                    
                    if not line_text or re.match(r'^(table\s+of\s+)?contents$', line_text, re.IGNORECASE):
                        continue
                    
                    page_match = re.search(r'[.\s]+(?:page\s+)?(\d+)\s*$', line_text, re.IGNORECASE)
                    
                    if page_match:
                        page_number = int(page_match.group(1))
                        title = re.sub(r'[.\s]+(?:page\s+)?\d+\s*$', '', line_text, flags=re.IGNORECASE).strip()
                        title = re.sub(r'\.{2,}', '', title).strip()
                        
                        if title:
                            if pending_header:
                                toc_entries.append({
                                    "title": pending_header["title"],
                                    "page": page_number,
                                    "indent": pending_header["indent"],
                                    "font_size": pending_header["font_size"],
                                    "original_text": pending_header["original_text"],
                                    "is_chapter_header": True
                                })
                                pending_header = None
                            
                            toc_entries.append({
                                "title": title,
                                "page": page_number,
                                "indent": line_x,
                                "font_size": round(font_size, 2),
                                "original_text": line_text
                            })
                    else:
                        if re.match(r'chapter\s+\d+', line_text, re.IGNORECASE) or font_size > 14.5:
                            title = re.sub(r'\s+', ' ', line_text).strip()
                            title = re.sub(r':\s*$', '', title)
                            
                            pending_header = {
                                "title": title,
                                "indent": line_x,
                                "font_size": round(font_size, 2),
                                "original_text": line_text
                            }
    doc.close()
    
    if toc_entries:
        print(f"✓ Extracted {len(toc_entries)} entries from TOC page")
        return toc_entries
    
    return toc_entries


def get_visual_lines(page):
    """
    Reconstruct visual lines from a page by clustering text segments based on vertical overlap.
    Returns a list of dicts, where each dict represents a full visual line with 'text' and 'x0'.
    """
    blocks = page.get_text("dict")["blocks"]
    segments = []
    
    # 1. Flatten all text segments from all blocks
    for block in blocks:
        if "lines" in block:
            for line in block["lines"]:
                text = "".join(s["text"] for s in line["spans"]).strip()
                if text:
                    segments.append({
                        "text": text,
                        "bbox": line["bbox"],
                        "height": line["bbox"][3] - line["bbox"][1]
                    })
    
    # 2. Sort segments by vertical position (top edge)
    segments.sort(key=lambda s: s["bbox"][1])
    
    # 3. Cluster into visual lines
    visual_lines = []
    
    for seg in segments:
        # Try to find an existing line this segment belongs to
        matched = False
        for v_line in visual_lines:
            # Check vertical overlap
            line_top = v_line["bbox"][1]
            line_bottom = v_line["bbox"][3]
            seg_top = seg["bbox"][1]
            seg_bottom = seg["bbox"][3]
            
            overlap_top = max(line_top, seg_top)
            overlap_bottom = min(line_bottom, seg_bottom)
            overlap = max(0, overlap_bottom - overlap_top)
            
            # Check if overlap is significant relative to the segment height
            if overlap > 0.5 * seg["height"]:
                v_line["segments"].append(seg)
                # Update line bbox to include this segment
                v_line["bbox"] = (
                    min(v_line["bbox"][0], seg["bbox"][0]),
                    min(v_line["bbox"][1], seg["bbox"][1]),
                    max(v_line["bbox"][2], seg["bbox"][2]),
                    max(v_line["bbox"][3], seg["bbox"][3])
                )
                matched = True
                break
        
        if not matched:
            # Start a new visual line
            visual_lines.append({
                "segments": [seg],
                "bbox": seg["bbox"]
            })
    
    # 4. Sort lines by Y-position
    visual_lines.sort(key=lambda l: l["bbox"][1])
    
    # 5. Construct text for each line
    final_lines = []
    for v_line in visual_lines:
        # Sort segments in the line by X-position
        v_line["segments"].sort(key=lambda s: s["bbox"][0])
        
        # Join text
        full_text = " ".join(s["text"] for s in v_line["segments"])
        final_lines.append({
            "text": full_text,
            "x0": v_line["bbox"][0], # Keep indentation info
            "bbox": v_line["bbox"],
            "segments": v_line["segments"]
        })
        
    return final_lines


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


def extract_toc_ocr_style(pdf_path, toc_page_nums):
    """
    Extract TOC from OCR'd PDFs where titles and page numbers are on separate lines.
    This handles cases where the TOC has all titles first, then all page numbers.
    Also captures x-position (indentation) for hierarchy detection.
    """
    doc = fitz.open(pdf_path)
    page_count = doc.page_count
    
    all_lines = []
    
    # Collect all text lines from TOC pages with layout information
    for toc_page_num in toc_page_nums:
        page = doc[toc_page_num]
        # Use visual line reconstruction instead of raw blocks
        visual_lines = get_visual_lines(page)
        all_lines.extend(visual_lines)
    
    doc.close()
    
    # Skip the "TABLE OF CONTENTS" header
    all_lines = [l for l in all_lines if not re.match(r'^(table\s+of\s+)?contents$', l["text"], re.IGNORECASE)]
    
    # Separate potential titles from page numbers
    # Use a state machine approach to handle multi-line titles and split lines
    toc_entries = []
    current_title_lines = []
    current_x0 = None
    last_y_bottom = None
    
    for line in all_lines:
        text = line["text"]
        x0 = line["x0"]
        bbox = line.get("bbox")
        segments = line.get("segments", [])
        
        # Check for page number
        page_num = None
        title_part = None
        
        # 1. Standalone page number
        cleaned = clean_ocr_page_num(text)
        if cleaned is not None:
            if cleaned <= page_count + 50:
                page_num = cleaned
            # If it's a number but invalid (too large), we treat it as garbage and ignore it
            # We do NOT treat it as a title
        else:
            # 2. Trailing page number
            # Use segments if available to check for spatial separation
            page_num_candidate = None
            title_candidate = None
            
            if segments and len(segments) > 1:
                # Check the last segment
                last_seg = segments[-1]
                last_text = last_seg["text"].strip()
                
                # Check if last segment looks like a page number
                pot_num = clean_ocr_page_num(last_text)
                if pot_num is not None:
                    # Check separation from previous segment
                    prev_seg = segments[-2]
                    gap = last_seg["bbox"][0] - prev_seg["bbox"][2]
                    
                    # If gap is significant (> 10) OR text is clearly a number
                    if gap > 10 or last_text.isdigit():
                        page_num_candidate = pot_num
                        # Title is everything before
                        title_candidate = " ".join(s["text"] for s in segments[:-1])
            
            if page_num_candidate is not None:
                 if page_num_candidate <= page_count + 50:
                    page_num = page_num_candidate
                    title_part = title_candidate
            else:
                # Fallback to regex
                match = re.search(r'^(.*)\s+([0-9IlOoSBZ]{1,4})$', text)
                if match:
                    pot_num = clean_ocr_page_num(match.group(2))
                    if pot_num is not None and pot_num <= page_count + 50:
                        # Only accept if the "number" part doesn't look like a word part
                        # e.g. "Analysis" -> "Analys" "is" (15) - reject
                        # "Chapter I" -> "Chapter" "I" (1) - accept
                        
                        suffix = match.group(2)
                        prefix = match.group(1)
                        
                        is_valid_num = True
                        if not suffix.isdigit() and len(suffix) >= 2:
                             # If it's letters, check if it's a common word ending
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
            # Check if title starts with a number BEFORE stripping it
            # This helps identify new numbered sections that should reset previous accumulated text
            starts_with_number = bool(re.match(r'^\d', title_part.strip()))
            
            # Clean title part (fix wide spacing like "B I G B I L L")
            # Replace "Letter Space" with "Letter" if followed by another letter
            title_part = re.sub(r'\b([A-Z])\s+(?=[A-Z]\b)', r'\1', title_part)
            
            # Strip leading numbering (e.g. "1. ", "4. ")
            title_part = re.sub(r'^\d+\.?\s*', '', title_part)
            
            # Validate title_part
            is_valid_part = True
            
            # Filter out separators
            if re.match(r'^[—_\-\s]+$|^[a-z]{1,3}\s+[A-Z][a-z]\s+', title_part):
                is_valid_part = False
            
            # If starting a new entry, apply stricter checks
            if not current_title_lines and is_valid_part:
                word_count = len(title_part.split())
                if word_count == 1:
                    common_words = {'PART', 'LOGIC', 'WRITING', 'CHAPTER', 'SECTION', '|'}
                    # Allow shorter words if they look like titles (e.g. "Index", "Notes")
                    if not (title_part[0].isupper() and len(title_part) >= 3 and title_part not in common_words):
                        is_valid_part = False
                elif word_count > 1:
                    # Check if it looks like text (not just symbols)
                    if not any(c.isalpha() for c in title_part):
                        is_valid_part = False
            
            if is_valid_part:
                # Check vertical gap to detect if we should reset
                # This handles cases where we have headers like "PART 1" (no page num)
                # followed by a large gap before the next item.
                if bbox and last_y_bottom is not None:
                    this_y_top = bbox[1]
                    gap = this_y_top - last_y_bottom
                    # If gap is large (e.g. > 20 units), assume the previous lines were orphaned headers
                    if gap > 20:
                        current_title_lines = []
                        current_x0 = None
                
                # Check if this line starts with a number (indicating a new section)
                # while we already have accumulated text that doesn't look like a prefix.
                if current_title_lines and starts_with_number:
                    # Check if previous lines look like a prefix
                    prev_text = " ".join(current_title_lines).upper()
                    is_prefix = False
                    for prefix in ['CHAPTER', 'PART', 'SECTION', 'UNIT', 'MODULE']:
                        if prev_text.endswith(prefix) or prev_text == prefix:
                            is_prefix = True
                            break
                    
                    if not is_prefix:
                        current_title_lines = []
                        current_x0 = None
                
                if not current_title_lines:
                    current_x0 = x0
                current_title_lines.append(title_part)
                
                if bbox:
                    last_y_bottom = bbox[3]
            
        if page_num is not None:
            # We have a page number. This completes the entry.
            if current_title_lines:
                full_title = " ".join(current_title_lines)
                toc_entries.append({
                    "title": full_title,
                    "page": page_num,
                    "indent": current_x0,
                    "font_size": 12.0,
                    "original_text": full_title
                })
                # Reset
                current_title_lines = []
                current_x0 = None
                last_y_bottom = None
    
    print(f"✓ Matched {len(toc_entries)} entries using OCR-style extraction")
    
    return toc_entries


def assign_toc_hierarchy(toc_entries, use_ocr_mode=False):
    """Assign hierarchy levels based on indentation and font size."""
    if not toc_entries:
        return []
    
    indents = sorted(set(entry["indent"] for entry in toc_entries))
    
    # Check if we have x-position based indents (large values > 100)
    # These come from OCR mode and need clustering
    # ONLY apply this clustering in OCR mode, as standard PDFs have precise coordinates
    if use_ocr_mode and indents and max(indents) > 100:
        # Cluster similar x-positions within 0.5 units to handle OCR noise
        # while preserving small indentations (like 1.0 unit)
        clustered = []
        for indent in indents:
            # Find existing cluster within 0.5 units
            found_cluster = None
            for cluster in clustered:
                if abs(indent - cluster) <= 0.5:
                    found_cluster = cluster
                    break
            if found_cluster is None:
                clustered.append(indent)
        indents = sorted(clustered)
    
    font_sizes = sorted(set(entry["font_size"] for entry in toc_entries), reverse=True)
    
    for entry in toc_entries:
        # For clustered indents, find the closest cluster
        if use_ocr_mode and indents and max(indents) > 100:
            closest_indent = min(indents, key=lambda x: abs(x - entry["indent"]))
            indent_score = indents.index(closest_indent)
        else:
            indent_score = indents.index(entry["indent"])
        
        font_score = font_sizes.index(entry["font_size"])
        entry["hierarchy_score"] = indent_score + (font_score * 0.5)
    
    scores = sorted(set(entry["hierarchy_score"] for entry in toc_entries))
    
    for entry in toc_entries:
        entry["level"] = scores.index(entry["hierarchy_score"]) + 1
    
    return toc_entries


def find_heading_in_document(pdf_path, heading_title, expected_page, search_window=5, exclude_pages=None, ocr_mode=False):
    """Find where a heading actually appears in the document."""
    doc = fitz.open(pdf_path)
    page_count = len(doc)
    
    # Convert exclude_pages to 0-indexed set
    exclude_page_nums = set()
    if exclude_pages:
        exclude_page_nums = {p - 1 if p > 0 else p for p in exclude_pages}
    
    # Standard mode (and OCR fallback): use fuzzy matching with search window
    normalized_title = re.sub(r'\s+', ' ', heading_title.lower().strip())
    normalized_title = re.sub(r'[^\w\s]', '', normalized_title)
    title_nospace = normalized_title.replace(" ", "")
    
    # If expected page is out of range, search the entire document
    if expected_page < 1 or expected_page > page_count:
        start_page = 0
        end_page = page_count
    else:
        start_page = max(0, expected_page - search_window)
        end_page = min(page_count, expected_page + search_window)
    
    best_match_page = min(expected_page, page_count)
    best_match_score = 0
    
    for page_num in range(start_page, end_page):
        # Skip TOC pages
        if page_num in exclude_page_nums:
            continue
        page = doc[page_num]
        text = page.get_text().lower()
        
        normalized_text = re.sub(r'\s+', ' ', text)
        normalized_text = re.sub(r'[^\w\s]', '', normalized_text)
        
        # Check 1: Standard substring match
        if normalized_title in normalized_text:
            doc.close()
            return page_num + 1, 1.0  # Perfect match
            
        # Check 2: Ignore spaces match (handles "BIGBILL" vs "BIG BILL")
        if title_nospace and len(title_nospace) > 4 and title_nospace in normalized_text.replace(" ", ""):
            doc.close()
            return page_num + 1, 0.95  # Near perfect match
        
        title_words = set(normalized_title.split())
        text_words = set(normalized_text.split())
        
        if not title_words:
            continue
            
        # Calculate match score
        matching_words = title_words & text_words
        match_score = len(matching_words) / len(title_words)
        
        # For better matching, require at least 3 matching words OR 80% match for short titles
        # This prevents matching common words like "INTRODUCTION TO PART ONE" when only scattered words match
        min_words_matched = min(3, len(title_words))
        if len(matching_words) < min_words_matched and match_score < 0.8:
            match_score = match_score * 0.7  # Penalize matches with few words
        
        if match_score > best_match_score:
            best_match_score = match_score
            best_match_page = page_num + 1
            
    doc.close()
    
    # Use a lower threshold if we had to search the entire document
    threshold = 0.5 if (expected_page < 1 or expected_page > page_count) else 0.7
    
    if best_match_score > threshold:
        return best_match_page, best_match_score
    
    return min(expected_page, page_count), best_match_score


def calculate_page_offset(pdf_path, toc_entries, exclude_pages=None):
    """
    Calculate the page offset between TOC page numbers and PDF page numbers.
    Uses a sample of long, unique titles to find their actual locations in the PDF.
    """
    print("Calculating page offset...")
    
    # Filter for good candidates: long titles (more likely unique)
    candidates = [e for e in toc_entries if len(e["title"]) > 15]
    # Sort by length descending
    candidates.sort(key=lambda x: len(x["title"]), reverse=True)
    # Take top 10
    candidates = candidates[:10]
    
    if not candidates:
        return 0
        
    offsets = []
    
    for entry in candidates:
        # Search entire document for this entry
        # We use find_heading_in_document but force it to search everywhere by passing expected_page=-1
        actual_page, score = find_heading_in_document(
            pdf_path, 
            entry["title"], 
            -1, # Force full search
            exclude_pages=exclude_pages,
            ocr_mode=True # Use relaxed matching logic
        )
        
        if score > 0.9: # High confidence match
            offset = actual_page - entry["page"]
            offsets.append(offset)
            print(f"  Found offset {offset} for '{entry['title']}' (Page {entry['page']} -> {actual_page})")
            
    if not offsets:
        print("  Could not determine offset (no high-confidence matches found)")
        return 0
        
    # Return median offset
    offsets.sort()
    median_offset = offsets[len(offsets) // 2]
    print(f"✓ Detected median page offset: {median_offset}")
    return median_offset


def match_toc_to_document(pdf_path, toc_entries, toc_pages=None, ocr_mode=False, approximate_missing=False):
    """Match TOC entries to actual locations in the document."""
    matched_entries = []
    skipped_entries = []
    all_entries_with_status = []  # Track all entries with found/not found status
    
    # Convert toc_pages to 1-indexed list for exclude_pages
    # Also exclude the first few pages (likely copyright, title page, etc.) to avoid false matches
    if toc_pages:
        first_toc_page = min(toc_pages) + 1  # +1 for 1-indexed
        # Exclude pages 1-3 (typical frontmatter) and the TOC pages themselves
        exclude_pages = list(range(1, min(4, first_toc_page))) + [p + 1 for p in toc_pages]
    else:
        exclude_pages = None
        
    # Calculate global page offset
    page_offset = calculate_page_offset(pdf_path, toc_entries, exclude_pages)
    
    print(f"\nMatching {len(toc_entries)} TOC entries to document...")
    
    for i, entry in enumerate(toc_entries):
        # Apply offset to expected page
        expected_page = entry["page"] + page_offset
        
        actual_page, match_score = find_heading_in_document(
            pdf_path, 
            entry["title"], 
            expected_page,
            exclude_pages=exclude_pages,
            ocr_mode=ocr_mode
        )
        
        # Store entry with its search result
        entry_with_status = {
            "title": entry["title"],
            "page": actual_page,
            "level": entry["level"],
            "toc_page": entry["page"],
            "match_score": match_score,
            "found": False
        }
        
        # In OCR mode, check if exact match was found
        # Relaxed threshold for OCR mode since we now allow fuzzy matching
        threshold = 0.7 if ocr_mode else 0.85
        
        if match_score < threshold:
            skipped_entries.append({"title": entry["title"], "score": match_score, "index": i})
            all_entries_with_status.append(entry_with_status)
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(toc_entries)} entries...")
            continue
        
        matched_entry = {
            "title": entry["title"],
            "page": actual_page,
            "level": entry["level"],
            "toc_page": entry["page"],
            "match_score": match_score,
            "indent": entry.get("indent", 0),
            "font_size": entry.get("font_size", 12.0)
        }
        
        entry_with_status["found"] = True
        all_entries_with_status.append(entry_with_status)
        matched_entries.append(matched_entry)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(toc_entries)} entries...")
    
    if skipped_entries:
        print(f"⚠ Skipped {len(skipped_entries)} entries with poor matches (not found in document)")
        if len(skipped_entries) <= 5:
            for entry in skipped_entries:
                print(f"  - {entry['title']} (match score: {entry['score']:.2f})")
    
    print(f"✓ Matched {len(matched_entries)} entries to document pages")
    
    # If approximate_missing is enabled and we have some matched entries, approximate the missing ones
    if approximate_missing and matched_entries and skipped_entries:
        approximated = approximate_missing_entries(all_entries_with_status, matched_entries, skipped_entries, pdf_path)
        if approximated:
            print(f"ℹ Approximated {len(approximated)} missing entries (marked with ★)")
            # Merge approximated entries back into their correct TOC positions
            # Rebuild the full list in original TOC order
            full_list = []
            # Map by original title (without star marker)
            approx_dict = {}
            for e in approximated:
                original_title = e["title"].replace(" ★", "")
                approx_dict[original_title] = e
            
            for entry in all_entries_with_status:
                if entry["found"]:
                    # Find the corresponding matched entry
                    matched = next((m for m in matched_entries if m["title"] == entry["title"]), None)
                    if matched:
                        full_list.append(matched)
                elif entry["title"] in approx_dict:
                    # Use the approximated entry
                    full_list.append(approx_dict[entry["title"]])
            
            matched_entries = full_list
    
    return matched_entries


def approximate_missing_entries(all_entries, matched_entries, skipped_entries, pdf_path=None):
    """Approximate page numbers for entries that weren't found based on nearby matches.
    
    Uses TOC page numbers to understand spacing between sections and makes intelligent
    approximations. Skips entries that would be beyond the PDF page count.
    """
    approximated = []
    
    # Get PDF page count to avoid approximating beyond it
    pdf_page_count = None
    if pdf_path:
        import fitz
        doc = fitz.open(pdf_path)
        pdf_page_count = doc.page_count
        doc.close()
    
    # Create a map of found entries by their index in the original list
    found_map = {}
    toc_page_map = {}  # Map of index to TOC page number
    for i, entry in enumerate(all_entries):
        toc_page_map[i] = entry.get("toc_page", 0)
        if entry["found"]:
            found_map[i] = entry["page"]
    
    if not found_map:
        return approximated
    
    # For each skipped entry, find nearest found entries and interpolate
    for skip_info in skipped_entries:
        idx = skip_info["index"]
        entry = all_entries[idx]
        toc_page = entry.get("toc_page", 0)
        
        # Find the nearest found entry before and after this index
        before_idx = None
        before_page = None
        before_toc_page = None
        for i in range(idx - 1, -1, -1):
            if i in found_map:
                before_idx = i
                before_page = found_map[i]
                before_toc_page = toc_page_map.get(i, 0)
                break
        
        after_idx = None
        after_page = None
        after_toc_page = None
        for i in range(idx + 1, len(all_entries)):
            if i in found_map:
                after_idx = i
                after_page = found_map[i]
                after_toc_page = toc_page_map.get(i, 0)
                break
        
        # ONLY approximate entries that fall between two found entries (interpolation)
        # Skip extrapolation to avoid creating bookmarks to non-existent content
        approx_page = None
        
        if before_idx is not None and after_idx is not None:
            # We have found entries both before and after - safe to interpolate
            # Use TOC page differences to guide interpolation
            toc_diff_total = after_toc_page - before_toc_page if after_toc_page and before_toc_page else 0
            toc_diff_to_here = toc_page - before_toc_page if toc_page and before_toc_page else 0
            
            if toc_diff_total > 0 and toc_diff_to_here > 0:
                # Use TOC spacing ratio for intelligent interpolation
                pdf_page_diff = after_page - before_page
                ratio = toc_diff_to_here / toc_diff_total
                approx_page = before_page + int(pdf_page_diff * ratio)
            else:
                # Fall back to simple linear interpolation by index
                idx_diff = after_idx - before_idx
                page_diff = after_page - before_page
                position = idx - before_idx
                approx_page = before_page + int((page_diff * position) / idx_diff)
        else:
            # Cannot approximate - need entries both before AND after for safe interpolation
            # Skip extrapolation to avoid false bookmarks
            continue
        
        # Validate the approximation
        if approx_page and approx_page > 0:
            # Skip if approximation exceeds PDF page count
            if pdf_page_count and approx_page > pdf_page_count:
                continue
                
            # Skip if approximation is not reasonable (too far from found entries)
            if before_page and approx_page > before_page + 10:  # More than 10 pages away seems unreasonable
                continue
            if after_page and approx_page < after_page - 10:
                continue
            
            # Check if this page conflicts with an already-found entry
            occupied_pages = set(found_map.values())
            if approx_page in occupied_pages and entry.get("level", 1) == 1:
                # Try to find a nearby unoccupied page
                if approx_page + 1 not in occupied_pages and (not pdf_page_count or approx_page + 1 <= pdf_page_count):
                    approx_page = approx_page + 1
                elif approx_page - 1 not in occupied_pages and approx_page - 1 >= 1:
                    approx_page = approx_page - 1
                else:
                    # Can't find a good alternative, skip this approximation
                    continue
            
            # VALIDATE: Search the approximated page for the entry title
            # Only keep approximation if we find similar text on that page
            validation_score = 0.0
            if pdf_path:
                import fitz
                doc = fitz.open(pdf_path)
                if 0 <= approx_page - 1 < len(doc):
                    page_text = doc[approx_page - 1].get_text()
                    # Normalize and check for fuzzy match
                    normalized_title = ' '.join(entry["title"].strip().split()).lower()
                    normalized_page = ' '.join(page_text.split()).lower()
                    
                    # Check if title appears in page text (fuzzy)
                    if normalized_title in normalized_page:
                        validation_score = 1.0
                    else:
                        # Try partial match - at least 50% of words should match
                        title_words = set(normalized_title.split())
                        page_words = set(normalized_page.split())
                        if title_words and page_words:
                            overlap = len(title_words & page_words) / len(title_words)
                            validation_score = overlap
                doc.close()
            
            # Only keep approximation if validation score is reasonable (>= 0.5)
            if validation_score < 0.5:
                continue
            
            approximated.append({
                "title": entry["title"] + " ★",  # Add star to indicate approximation
                "page": approx_page,
                "level": entry["level"],
                "toc_page": entry["toc_page"],
                "match_score": validation_score,
                "approximated": True,
                "indent": entry.get("indent", 0),
                "font_size": entry.get("font_size", 12.0)
            })
    
    return approximated


def extract_text_with_formatting(pdf_path):
    """Extract text from PDF with font size, style, and position information."""
    doc = fitz.open(pdf_path)
    text_blocks = []
    page_count = len(doc)
    
    for page_num in range(page_count):
        page = doc[page_num]
        blocks = page.get_text("dict", flags=11)["blocks"]
        
        for block in blocks:
            if block["type"] == 0:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if text:
                            text_blocks.append({
                                "text": text,
                                "page": page_num + 1,
                                "font_size": round(span["size"], 2),
                                "font_name": span["font"],
                                "flags": span["flags"],
                                "is_bold": bool(span["flags"] & 2**4),
                                "is_italic": bool(span["flags"] & 2**1),
                                "color": span["color"],
                                "bbox": span["bbox"]
                            })
    
    doc.close()
    print(f"Extracted {len(text_blocks)} text blocks from {page_count} pages")
    return text_blocks


def analyze_font_statistics(text_blocks):
    """Analyze font sizes and styles to determine body text vs. headings."""
    font_sizes = [block["font_size"] for block in text_blocks]
    font_size_counter = Counter(font_sizes)
    
    most_common_sizes = font_size_counter.most_common(10)
    body_text_size = most_common_sizes[0][0] if most_common_sizes else 12.0
    
    unique_sizes = sorted(set(font_sizes), reverse=True)
    
    stats = {
        "body_text_size": body_text_size,
        "unique_sizes": unique_sizes,
        "size_distribution": dict(most_common_sizes),
        "min_size": min(font_sizes) if font_sizes else 0,
        "max_size": max(font_sizes) if font_sizes else 0,
    }
    
    print(f"Body text size (most common): {body_text_size}")
    print(f"Font size range: {stats['min_size']} - {stats['max_size']}")
    print(f"Unique font sizes: {len(unique_sizes)}")
    
    return stats


def identify_headings(text_blocks, stats, min_heading_size_ratio=1.1):
    """Identify potential headings based on font size and formatting."""
    body_size = stats["body_text_size"]
    headings = []
    
    def is_likely_heading(block):
        text = block["text"]
        size = block["font_size"]
        
        if size < body_size * min_heading_size_ratio:
            return False
        
        if len(text) > 200:
            return False
        
        if len(text) > 50 and text.endswith('.'):
            return False
        
        if block["is_bold"]:
            return True
        
        if size >= body_size * 1.3:
            return True
        
        return False
    
    potential_headings = [block for block in text_blocks if is_likely_heading(block)]
    
    if potential_headings:
        heading_sizes = sorted(set([h["font_size"] for h in potential_headings]), reverse=True)
        
        size_to_level = {}
        for idx, size in enumerate(heading_sizes[:6]):
            size_to_level[size] = idx + 1
        
        for block in potential_headings:
            level = size_to_level.get(block["font_size"], 3)
            headings.append({
                "text": block["text"],
                "page": block["page"],
                "level": level,
                "font_size": block["font_size"],
                "is_bold": block["is_bold"]
            })
    
    print(f"Identified {len(headings)} potential headings")
    if headings:
        level_counts = Counter([h["level"] for h in headings])
        for level in sorted(level_counts.keys()):
            print(f"  Level {level} (H{level}): {level_counts[level]} headings")
    
    return headings


def generate_toc_structure(headings):
    """Generate a hierarchical table of contents structure."""
    if not headings:
        return []
    
    # Sort headings by page number to ensure proper bookmark order
    # Bookmarks MUST be in ascending page order in the PDF
    sorted_headings = sorted(headings, key=lambda x: x["page"])
    
    # Compress levels to consecutive integers starting from 1
    # This handles cases where filtering removed intermediate levels
    unique_levels = sorted(set(h["level"] for h in sorted_headings))
    level_map = {old: new for new, old in enumerate(unique_levels, 1)}
    
    toc = []
    prev_level = 0
    prev_page = 0
    
    for heading in sorted_headings:
        # Skip if page number doesn't increase (invalid bookmark order)
        if heading["page"] <= prev_page:
            continue
            
        normalized_level = level_map[heading["level"]]
        normalized_level = max(1, normalized_level)
        
        toc.append([
            normalized_level,
            heading["text"],
            heading["page"]
        ])
        
        prev_level = normalized_level
        prev_page = heading["page"]
    
    return toc


def normalize_toc_hierarchy(toc):
    """Ensure TOC hierarchy follows PyMuPDF rules."""
    if not toc:
        return []
    
    normalized = []
    prev_level = 0
    
    for entry in toc:
        level, title, page = entry[0], entry[1], entry[2]
        
        if not normalized:
            level = 1
        elif level > prev_level + 1:
            level = prev_level + 1
        level = max(1, level)
        
        normalized.append([level, title, page])
        prev_level = level
    
    return normalized


def add_toc_to_pdf(input_pdf_path, output_pdf_path, toc):
    """Add table of contents bookmarks to the PDF."""
    doc = fitz.open(input_pdf_path)
    
    # Validate page numbers
    page_count = doc.page_count
    for i, entry in enumerate(toc):
        if entry[2] > page_count:
            print(f"⚠ Warning: Entry {i} '{entry[1]}' has page {entry[2]} but PDF only has {page_count} pages. Adjusting to page {page_count}.")
            entry[2] = page_count
        elif entry[2] < 1:
            print(f"⚠ Warning: Entry {i} '{entry[1]}' has invalid page {entry[2]}. Adjusting to page 1.")
            entry[2] = 1
    
    doc.set_toc(toc)
    doc.save(output_pdf_path, garbage=4, deflate=True)
    doc.close()
    
    print(f"✓ Added {len(toc)} bookmarks to PDF")
    print(f"✓ Saved output to: {output_pdf_path}")


def generate_pdf_toc(input_pdf_path, output_pdf_path=None, 
                     use_existing_toc=True, flat_structure=False,
                     add_toc_bookmark=True, use_ocr_mode=False, approximate_missing=None):
    """Complete workflow to generate TOC for a PDF."""
    print("=" * 80)
    print("PDF TABLE OF CONTENTS GENERATOR")
    print("=" * 80)
    print(f"\nProcessing: {input_pdf_path}\n")
    
    if output_pdf_path is None:
        path = Path(input_pdf_path)
        output_pdf_path = str(path.parent / f"{path.stem}_with_toc{path.suffix}")
    
    # Default: enable approximation in OCR mode if not explicitly set
    if approximate_missing is None:
        approximate_missing = use_ocr_mode
    
    headings = None
    method_used = None
    toc_page_nums = None
    
    # Try to extract from existing TOC page first
    if use_existing_toc:
        print("Step 1: Checking for existing TOC page...")
        toc_page = detect_toc_page(input_pdf_path)
        print()
        
        if toc_page is not None:
            toc_page_nums = toc_page if isinstance(toc_page, list) else [toc_page]
            print("Step 2: Extracting TOC from dedicated page...")
            if use_ocr_mode:
                print("  (OCR mode enabled - will use enhanced extraction for poor quality scans)")
            toc_entries = extract_toc_from_page(input_pdf_path, toc_page, use_ocr_mode)
            
            if toc_entries:
                print("Step 3: Assigning hierarchy to TOC entries...")
                toc_entries = assign_toc_hierarchy(toc_entries, use_ocr_mode)
                
                print("Step 4: Matching TOC entries to document pages...")
                matched_entries = match_toc_to_document(input_pdf_path, toc_entries, toc_page_nums, use_ocr_mode, approximate_missing)
                
                headings = matched_entries
                method_used = "existing_toc"
                print(f"✓ Using TOC page method: {len(headings)} entries\n")
    
    # Fallback to font-based detection if no TOC found
    if headings is None:
        print("Step 1: Extracting text with formatting...")
        text_blocks = extract_text_with_formatting(input_pdf_path)
        print()
        
        print("Step 2: Analyzing font statistics...")
        stats = analyze_font_statistics(text_blocks)
        print()
        
        print("Step 3: Identifying headings...")
        headings = identify_headings(text_blocks, stats)
        print()
        
        if not headings:
            print("⚠ Warning: No headings found.")
            return None
        
        method_used = "font_detection"
        print(f"✓ Using font detection method: {len(headings)} entries\n")
    
    # Generate TOC structure
    print(f"Step {5 if method_used == 'existing_toc' else 4}: Generating TOC structure...")
    
    if method_used == "existing_toc":
        # Compress levels to consecutive integers starting from 1
        # This handles cases where filtering or sparse levels (OCR) created gaps
        unique_levels = sorted(set(h["level"] for h in headings))
        level_map = {old: new for new, old in enumerate(unique_levels, 1)}
        
        toc = []
        
        # Determine level offset
        # If we're adding a TOC bookmark, shift everything down by 1 level
        # so they become children of the TOC (or siblings if normalized later)
        level_offset = 1 if toc_page_nums and add_toc_bookmark else 0
        
        prev_page = 0
        skipped_count = 0
        
        for entry in headings:
            page = entry["page"]
            
            # Skip if page decreases (violates bookmark ordering requirement)
            # Note: Multiple bookmarks on the same page are allowed
            if page < prev_page:
                skipped_count += 1
                continue
            
            # Apply level compression and offset
            level = level_map[entry["level"]] + level_offset
            
            toc.append([
                level,
                entry["title"],
                page
            ])
            
            prev_page = page
        
        if skipped_count > 0:
            print(f"  ⚠ Skipped {skipped_count} entries that would violate page order")
        
        # Add TOC bookmark if requested, inserting it in the correct position
        if toc_page_nums and add_toc_bookmark:
            toc_page = toc_page_nums[0] + 1
            # Find where to insert TOC bookmark to maintain page order
            insert_pos = 0
            for i, entry in enumerate(toc):
                if entry[2] < toc_page:
                    insert_pos = i + 1
                else:
                    break
            toc.insert(insert_pos, [1, "Table of Contents", toc_page])
            
        toc = normalize_toc_hierarchy(toc)
    else:
        toc = generate_toc_structure(headings)
    
    # Apply flat structure if requested
    if flat_structure:
        print("✓ Converting to flat structure (all entries at level 1)")
        toc = [[1, entry[1], entry[2]] for entry in toc]
    
    print(f"Generated TOC with {len(toc)} entries")
    print()
    
    # Add TOC to PDF
    step_num = 7 if method_used == "existing_toc" else 6
    print(f"Step {step_num}: Adding bookmarks to PDF...")
    add_toc_to_pdf(input_pdf_path, output_pdf_path, toc)
    print()
    
    print("=" * 80)
    print(f"✓ COMPLETE! (Method: {method_used})")
    print("=" * 80)
    
    return output_pdf_path


def get_user_input(prompt, default=None, valid_values=None):
    """Get user input with default value and validation."""
    if default is not None:
        prompt = f"{prompt} [{default}]: "
    else:
        prompt = f"{prompt}: "
    
    while True:
        response = input(prompt).strip()
        
        if not response and default is not None:
            return default
        
        if not response:
            print("Please provide a value.")
            continue
        
        if valid_values and response.lower() not in valid_values:
            print(f"Please enter one of: {', '.join(valid_values)}")
            continue
        
        return response


def interactive_mode():
    """Run the tool in interactive mode."""
    print("=" * 80)
    print("PDF TABLE OF CONTENTS GENERATOR - Interactive Mode")
    print("=" * 80)
    print()
    
    # Get PDF path
    pdf_path_input = get_user_input("Enter the path to your PDF file")
    
    # Clean up the path: expand user, remove quotes, handle escaped spaces
    pdf_path = pdf_path_input.strip()
    # Remove surrounding quotes if present
    if (pdf_path.startswith('"') and pdf_path.endswith('"')) or \
       (pdf_path.startswith("'") and pdf_path.endswith("'")):
        pdf_path = pdf_path[1:-1]
    # Replace escaped spaces with regular spaces
    pdf_path = pdf_path.replace('\\ ', ' ')
    # Expand ~ to home directory
    pdf_path = str(Path(pdf_path).expanduser())
    
    if not Path(pdf_path).exists():
        print(f"Error: File not found: {pdf_path}")
        print(f"(You entered: {pdf_path_input})")
        sys.exit(1)
    
    # Get output path
    default_output = str(Path(pdf_path).parent / f"{Path(pdf_path).stem}_with_toc.pdf")
    output_path = get_user_input(f"Enter output PDF path", default=default_output)
    
    # Get options
    use_existing = get_user_input(
        "Try to extract from existing TOC page? (y/n)", 
        default="y", 
        valid_values=["y", "n"]
    ).lower() == "y"
    
    use_ocr_mode = get_user_input(
        "Is this a poorly scanned/OCR'd PDF? (y/n)", 
        default="n", 
        valid_values=["y", "n"]
    ).lower() == "y"
    
    flat_structure = get_user_input(
        "Use flat structure (no hierarchy)? (y/n)", 
        default="n", 
        valid_values=["y", "n"]
    ).lower() == "y"
    
    add_toc_bookmark = get_user_input(
        "Add 'Table of Contents' bookmark? (y/n)", 
        default="y", 
        valid_values=["y", "n"]
    ).lower() == "y"
    
    print()
    
    # Generate TOC
    generate_pdf_toc(
        pdf_path,
        output_path,
        use_existing_toc=use_existing,
        flat_structure=flat_structure,
        add_toc_bookmark=add_toc_bookmark,
        use_ocr_mode=use_ocr_mode
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate table of contents bookmarks for PDF files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python pdf_toc_generator.py
  
  # With arguments
  python pdf_toc_generator.py input.pdf -o output.pdf
  
  # Flat structure (no hierarchy)
  python pdf_toc_generator.py input.pdf --flat
  
  # Skip existing TOC page detection
  python pdf_toc_generator.py input.pdf --no-existing-toc
        """
    )
    
    parser.add_argument(
        "input_pdf",
        nargs="?",
        help="Path to input PDF file"
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to output PDF file (default: input_with_toc.pdf)"
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Create flat structure (all bookmarks at level 1)"
    )
    parser.add_argument(
        "--no-existing-toc",
        action="store_true",
        help="Don't try to extract from existing TOC page"
    )
    parser.add_argument(
        "--no-toc-bookmark",
        action="store_true",
        help="Don't add 'Table of Contents' bookmark"
    )
    parser.add_argument(
        "--ocr-mode",
        action="store_true",
        help="Use OCR mode for poorly scanned PDFs (enhanced extraction for separated titles/page numbers)"
    )
    parser.add_argument(
        "--approximate",
        action="store_true",
        help="Approximate page numbers for TOC entries that can't be found exactly (useful for excerpts)"
    )
    
    args = parser.parse_args()
    
    # If no input PDF provided, run interactive mode
    if not args.input_pdf:
        interactive_mode()
    else:
        # Clean up the input path
        input_path = args.input_pdf.strip()
        if (input_path.startswith('"') and input_path.endswith('"')) or \
           (input_path.startswith("'") and input_path.endswith("'")):
            input_path = input_path[1:-1]
        input_path = input_path.replace('\\ ', ' ')
        input_path = str(Path(input_path).expanduser())
        
        if not Path(input_path).exists():
            print(f"Error: File not found: {input_path}")
            sys.exit(1)
        
        generate_pdf_toc(
            input_path,
            args.output,
            use_existing_toc=not args.no_existing_toc,
            flat_structure=args.flat,
            add_toc_bookmark=not args.no_toc_bookmark,
            use_ocr_mode=args.ocr_mode,
            approximate_missing=args.approximate if args.approximate else None
        )


if __name__ == "__main__":
    main()
