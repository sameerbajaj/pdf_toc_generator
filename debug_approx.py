# Simulate what should happen during approximation

entries = [
    {"title": "PREFACE", "toc_page": 4, "level": 1, "found": True, "doc_page": 4},  # idx 0
    {"title": "INTRODUCTION TO PART ONE", "toc_page": 7, "level": 1, "found": True, "doc_page": 5},  # idx 1
    {"title": "WHY A PYRAMID STRUCTURE?", "toc_page": 17, "level": 1, "found": False},  # idx 2
    {"title": "Sorting into Pyramids", "toc_page": 19, "level": 2, "found": False},  # idx 3
    {"title": "The Magical Number Seven", "toc_page": 20, "level": 2, "found": True, "doc_page": 20},  # idx 4
    {"title": "The Need to State the Logic", "toc_page": 21, "level": 2, "found": True, "doc_page": 21},  # idx 5
    {"title": "Thinking from the Bottom Up", "toc_page": 25, "level": 2, "found": False},  # idx 6
    {"title": "THE SUBSTRUCTURES WITHIN THE PYRAMID", "toc_page": 21, "level": 1, "found": False},  # idx 7
    {"title": "The Vertical Relationship", "toc_page": 24, "level": 2, "found": False},  # idx 8
    {"title": "The Horizontal Relationship", "toc_page": 26, "level": 2, "found": False},  # idx 9
]

print("Approximation analysis:")
print("\nFound entries (before/after markers):")
for idx, e in enumerate(entries):
    if e["found"]:
        print(f"  idx {idx}: {e['title']} → doc_page {e['doc_page']}, toc_page {e['toc_page']}")

print("\nApproximating missing entries:")
found_map = {idx: e["doc_page"] for idx, e in enumerate(entries) if e["found"]}
toc_page_map = {idx: e["toc_page"] for idx, e in enumerate(entries)}

for idx, entry in enumerate(entries):
    if entry["found"]:
        continue
    
    # Find before/after
    before_idx = before_page = before_toc = None
    for i in range(idx - 1, -1, -1):
        if i in found_map:
            before_idx = i
            before_page = found_map[i]
            before_toc = toc_page_map[i]
            break
    
    after_idx = after_page = after_toc = None
    for i in range(idx + 1, len(entries)):
        if i in found_map:
            after_idx = i
            after_page = found_map[i]
            after_toc = toc_page_map[i]
            break
    
    toc_page = entry["toc_page"]
    
    if before_idx is not None and after_idx is not None:
        toc_diff_total = after_toc - before_toc
        toc_diff_to_here = toc_page - before_toc
        if toc_diff_total > 0 and toc_diff_to_here > 0:
            pdf_page_diff = after_page - before_page
            ratio = toc_diff_to_here / toc_diff_total
            approx_page = before_page + int(pdf_page_diff * ratio)
        else:
            idx_diff = after_idx - before_idx
            page_diff = after_page - before_page
            position = idx - before_idx
            approx_page = before_page + int((page_diff * position) / idx_diff)
        
        print(f"\nidx {idx}: {entry['title']}")
        print(f"  before: idx {before_idx} (doc_page {before_page}, toc_page {before_toc})")
        print(f"  after: idx {after_idx} (doc_page {after_page}, toc_page {after_toc})")
        print(f"  this toc_page: {toc_page}")
        print(f"  toc_diff_total: {toc_diff_total}, toc_diff_to_here: {toc_diff_to_here}")
        print(f"  → approximated to: {approx_page}")

print("\n\nFiltering with page order:")
all_entries = []
for idx, e in enumerate(entries):
    if e["found"]:
        all_entries.append({"title": e["title"], "page": e["doc_page"], "level": e["level"]})
    else:
        # Check approximation...
        before_idx = before_page = before_toc = None
        for i in range(idx - 1, -1, -1):
            if i in found_map:
                before_idx = i
                before_page = found_map[i]
                before_toc = toc_page_map[i]
                break
        
        after_idx = after_page = after_toc = None
        for i in range(idx + 1, len(entries)):
            if i in found_map:
                after_idx = i
                after_page = found_map[i]
                after_toc = toc_page_map[i]
                break
        
        if before_idx is not None and after_idx is not None:
            toc_page = e["toc_page"]
            toc_diff_total = after_toc - before_toc
            toc_diff_to_here = toc_page - before_toc
            if toc_diff_total > 0 and toc_diff_to_here > 0:
                pdf_page_diff = after_page - before_page
                ratio = toc_diff_to_here / toc_diff_total
                approx_page = before_page + int(pdf_page_diff * ratio)
                
                # Check conflict
                occupied_pages = {e2["page"] for e2 in all_entries}
                if approx_page in occupied_pages and e["level"] == 1:
                    if approx_page + 1 not in occupied_pages and approx_page + 1 <= 26:
                        approx_page = approx_page + 1
                    elif approx_page - 1 not in occupied_pages and approx_page - 1 >= 1:
                        approx_page = approx_page - 1
                    else:
                        continue  # Skip
                
                all_entries.append({"title": e["title"], "page": approx_page, "level": e["level"]})

print("\nAll entries before page-order filtering:")
for e in all_entries:
    print(f"  L{e['level']}: {e['title']} → page {e['page']}")

print("\nAfter page-order filtering:")
prev_page = 0
filtered = []
for e in all_entries:
    if e["page"] <= prev_page:
        print(f"  SKIPPED (page {e['page']} <= {prev_page}): {e['title']}")
        continue
    filtered.append(e)
    prev_page = e["page"]

print("\nFinal entries:")
for e in filtered:
    print(f"  L{e['level']}: {e['title']} → page {e['page']}")
