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


def extract_toc_ocr_style(pdf_path, toc_page_nums):
    """
    Extract TOC from OCR'd PDFs where titles and page numbers are on separate lines.
    This handles cases where the TOC has all titles first, then all page numbers.
    """
    doc = fitz.open(pdf_path)
    
    all_lines = []
    
    # Collect all text lines from TOC pages
    for toc_page_num in toc_page_nums:
        page = doc[toc_page_num]
        text = page.get_text()
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        all_lines.extend(lines)
    
    doc.close()
    
    # Skip the "TABLE OF CONTENTS" header
    all_lines = [l for l in all_lines if not re.match(r'^(table\s+of\s+)?contents$', l, re.IGNORECASE)]
    
    # Separate potential titles from page numbers
    # Keep it simple: treat each line as either a title or page number
    titles = []
    page_numbers = []
    
    for line in all_lines:
        # Check if line is just a number (page number)
        if re.match(r'^\d+$', line):
            page_numbers.append(int(line))
        # Check if line is a separator or garbled text - skip it
        elif re.match(r'^[—_\-\s]+$|^[a-z]{1,3}\s+[A-Z][a-z]\s+', line):
            continue
        # Otherwise treat as title
        elif not line.isdigit():
            word_count = len(line.split())
            # For single words: only keep if they're ALL CAPS and look like section markers
            # Skip common words like "WRITING", "LOGIC", "PART"
            if word_count == 1:
                common_words = {'PART', 'LOGIC', 'WRITING', 'CHAPTER', 'SECTION'}
                if line.isupper() and len(line) >= 7 and line not in common_words:
                    titles.append(line)
            # For multi-word entries: keep if they look like headings
            # Either all caps (section headers) or title case (chapter names)
            elif word_count > 1:
                # Check if it's title case (first letter of each word capitalized)
                is_title_case = all(word[0].isupper() if word else False for word in line.split() if len(word) > 2)
                if line.isupper() or is_title_case:
                    titles.append(line)
    
    # Try to match titles to page numbers
    toc_entries = []
    
    if len(titles) > 0 and len(page_numbers) > 0:
        # Match titles to page numbers 1-to-1
        num_matches = min(len(titles), len(page_numbers))
        
        print(f"  Found {len(titles)} potential titles and {len(page_numbers)} page numbers")
        
        # Quick validation: check if page numbers are reasonable
        doc = fitz.open(pdf_path)
        page_count = doc.page_count
        doc.close()
        
        out_of_range_count = sum(1 for p in page_numbers[:num_matches] if p > page_count)
        
        if out_of_range_count > num_matches * 0.5:
            print(f"  ⚠ Warning: {out_of_range_count}/{num_matches} page numbers exceed PDF page count ({page_count})")
            print(f"  This appears to be an excerpt or the TOC references the original document's page numbers.")
            print(f"  Will use fuzzy matching to find headings in the available pages.")
        
        if num_matches > 0:
            for i in range(num_matches):
                toc_entries.append({
                    "title": titles[i],
                    "page": page_numbers[i],
                    "indent": 0,  # Can't determine from OCR
                    "font_size": 12.0,  # Default
                    "original_text": titles[i]
                })
            
            print(f"✓ Matched {len(toc_entries)} entries using OCR-style extraction")
    
    return toc_entries


def assign_toc_hierarchy(toc_entries):
    """Assign hierarchy levels based on indentation and font size."""
    if not toc_entries:
        return []
    
    indents = sorted(set(entry["indent"] for entry in toc_entries))
    font_sizes = sorted(set(entry["font_size"] for entry in toc_entries), reverse=True)
    
    for entry in toc_entries:
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
    
    # In OCR mode, do exact text search across entire document
    if ocr_mode:
        # Normalize the search title (collapse whitespace, case-insensitive)
        search_title = ' '.join(heading_title.strip().split()).lower()
        
        for page_num in range(page_count):
            # Skip TOC pages
            if page_num in exclude_page_nums:
                continue
            
            page = doc[page_num]
            text = page.get_text()
            
            # Normalize page text (collapse whitespace, case-insensitive)
            normalized_text = ' '.join(text.split()).lower()
            
            # Check for exact match
            if search_title in normalized_text:
                doc.close()
                return page_num + 1, 1.0  # Exact match found
        
        # No exact match found
        doc.close()
        return None, 0.0
    
    # Standard mode: use fuzzy matching with search window
    normalized_title = re.sub(r'\s+', ' ', heading_title.lower().strip())
    normalized_title = re.sub(r'[^\w\s]', '', normalized_title)
    
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
        
        if normalized_title in normalized_text:
            doc.close()
            return page_num + 1, 1.0  # Perfect match
        
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


def match_toc_to_document(pdf_path, toc_entries, toc_pages=None, ocr_mode=False):
    """Match TOC entries to actual locations in the document."""
    matched_entries = []
    skipped_entries = []
    
    # Convert toc_pages to 1-indexed list for exclude_pages
    exclude_pages = [p + 1 for p in toc_pages] if toc_pages else None
    
    print(f"\nMatching {len(toc_entries)} TOC entries to document...")
    
    for i, entry in enumerate(toc_entries):
        actual_page, match_score = find_heading_in_document(
            pdf_path, 
            entry["title"], 
            entry["page"],
            exclude_pages=exclude_pages,
            ocr_mode=ocr_mode
        )
        
        # In OCR mode, skip entries where no exact match was found
        if ocr_mode and (actual_page is None or match_score < 1.0):
            skipped_entries.append({"title": entry["title"], "score": match_score})
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(toc_entries)} entries...")
            continue
        
        # In non-OCR mode, use threshold of 0.85
        if not ocr_mode and match_score < 0.85:
            skipped_entries.append({"title": entry["title"], "score": match_score})
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(toc_entries)} entries...")
            continue
        
        matched_entry = {
            "title": entry["title"],
            "page": actual_page,
            "level": entry["level"],
            "toc_page": entry["page"],
            "match_score": match_score
        }
        
        matched_entries.append(matched_entry)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(toc_entries)} entries...")
    
    if skipped_entries:
        print(f"⚠ Skipped {len(skipped_entries)} entries with poor matches (not found in document)")
        if len(skipped_entries) <= 5:
            for entry in skipped_entries:
                print(f"  - {entry['title']} (match score: {entry['score']:.2f})")
    
    print(f"✓ Matched {len(matched_entries)} entries to document pages")
    
    return matched_entries


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
    
    first_level = sorted_headings[0]["level"]
    level_offset = 1 - first_level
    
    toc = []
    prev_level = 0
    prev_page = 0
    
    for heading in sorted_headings:
        # Skip if page number doesn't increase (invalid bookmark order)
        if heading["page"] <= prev_page:
            continue
            
        normalized_level = heading["level"] + level_offset
        normalized_level = max(1, normalized_level)
        
        if normalized_level > prev_level + 1:
            normalized_level = prev_level + 1
        
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
                     add_toc_bookmark=True, use_ocr_mode=False):
    """Complete workflow to generate TOC for a PDF."""
    print("=" * 80)
    print("PDF TABLE OF CONTENTS GENERATOR")
    print("=" * 80)
    print(f"\nProcessing: {input_pdf_path}\n")
    
    if output_pdf_path is None:
        path = Path(input_pdf_path)
        output_pdf_path = str(path.parent / f"{path.stem}_with_toc{path.suffix}")
    
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
                toc_entries = assign_toc_hierarchy(toc_entries)
                
                print("Step 4: Matching TOC entries to document pages...")
                matched_entries = match_toc_to_document(input_pdf_path, toc_entries, toc_page_nums, use_ocr_mode)
                
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
        toc = []
        
        # Build initial TOC, filtering out entries that would violate page order
        # Process in TOC order, but skip entries whose page number doesn't increase
        prev_page = 0
        skipped_count = 0
        
        for entry in headings:
            page = entry["page"]
            
            # Skip if page doesn't increase (violates bookmark ordering requirement)
            if page <= prev_page:
                skipped_count += 1
                continue
                
            toc.append([
                entry["level"],
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
            
            # Adjust levels of entries after TOC bookmark
            for i in range(len(toc)):
                if i != insert_pos:  # Don't adjust the TOC bookmark itself
                    toc[i][0] += 1
        
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
            use_ocr_mode=args.ocr_mode
        )


if __name__ == "__main__":
    main()
