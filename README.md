# PDF Table of Contents Generator

Automatically generate and add bookmarks (table of contents) to PDF files. Supports two methods:
1. **Extract from existing TOC page** - Intelligently detects and extracts from built-in TOC pages
2. **Font-based detection** - Analyzes text formatting to identify heading hierarchy

## Features

- 🔖 Automatically adds clickable bookmarks to PDFs
- 📄 Detects and extracts from existing TOC pages
- 🎯 Smart matching of TOC entries to document pages
- 🎛️ Multiple structure options (hierarchical or flat)
- ⚡ One-line installation and execution
- 🖥️ Interactive mode with guided prompts
- 🐍 Pure Python with minimal dependencies

## Quick Start

### One-Line Execution (with UV)

```bash
uvx --from git+https://github.com/sameerbajaj/pdf_toc_generator pdf_toc_generator
```

This will:
1. Install the tool automatically
2. Run in interactive mode
3. Guide you through the process with prompts

### Installation

#### Using UV (Recommended)

```bash
# Install UV if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Run the tool directly (no installation needed)
uvx --from git+https://github.com/sameerbajaj/pdf_toc_generator pdf_toc_generator
```

#### Manual Installation

```bash
git clone https://github.com/sameerbajaj/pdf_toc_generator.git
cd pdf_toc_generator
uv sync
uv run pdf_toc_generator.py
```

## Usage

### Interactive Mode (Recommended for First-Time Users)

Simply run without arguments to enter interactive mode:

```bash
uv run pdf_toc_generator.py
```

You'll be prompted for:
- PDF file path
- Output file path (optional, defaults to `input_with_toc.pdf`)
- Whether to extract from existing TOC page (y/n)
- Whether to use flat structure (y/n)
- Whether to add "Table of Contents" bookmark (y/n)

### Command-Line Arguments

```bash
# Basic usage
uv run pdf_toc_generator.py input.pdf

# Specify output file
uv run pdf_toc_generator.py input.pdf -o output.pdf

# Create flat structure (no hierarchy)
uv run pdf_toc_generator.py input.pdf --flat

# Skip existing TOC page detection (use font-based method)
uv run pdf_toc_generator.py input.pdf --no-existing-toc

# Don't add "Table of Contents" bookmark
uv run pdf_toc_generator.py input.pdf --no-toc-bookmark

# Use OCR mode for poorly scanned PDFs
uv run pdf_toc_generator.py scanned_document.pdf --ocr-mode

# Combine options
uv run pdf_toc_generator.py input.pdf -o output.pdf --flat --ocr-mode
```

### Help

```bash
uv run pdf_toc_generator.py --help
```

## Examples

### Example 1: Book with Existing TOC
```bash
$ uv run pdf_toc_generator.py book.pdf

================================================================================
PDF TABLE OF CONTENTS GENERATOR
================================================================================

Processing: book.pdf

Step 1: Checking for existing TOC page...
✓ Found TOC spanning pages 2-5

Step 2: Extracting TOC from dedicated page...
✓ Extracted 174 entries from TOC page
...
✓ COMPLETE! (Method: existing_toc)
```

### Example 2: Technical Document (Font-Based)
```bash
$ uv run pdf_toc_generator.py document.pdf --no-existing-toc

Processing: document.pdf

Step 1: Extracting text with formatting...
Extracted 1234 text blocks from 50 pages

Step 2: Analyzing font statistics...
Body text size (most common): 12.0
...
✓ COMPLETE! (Method: font_detection)
```

### Example 3: Flat Structure
```bash
$ uv run pdf_toc_generator.py report.pdf --flat -o report_bookmarks.pdf

✓ Converting to flat structure (all entries at level 1)
✓ Added 45 bookmarks to PDF
```

### Example 4: OCR Mode (Poorly Scanned PDFs)
```bash
$ uv run pdf_toc_generator.py scanned_book.pdf --ocr-mode

Step 2: Extracting TOC from dedicated page...
  (OCR mode enabled - will use enhanced extraction for poor quality scans)
  Found 25 potential titles and 25 page numbers
  ⚠ Warning: 20/25 page numbers exceed PDF page count (15)
  This appears to be an excerpt or the TOC references the original document's page numbers.
  Will use fuzzy matching to find headings in the available pages.
✓ Matched 25 entries using OCR-style extraction
...
✓ COMPLETE! (Method: existing_toc)
```

## How It Works

### Method 1: Existing TOC Page (Primary)

1. **Detects** TOC pages by looking for "Table of Contents", "Contents", or multiple chapter references
2. **Extracts** entries with page numbers, preserving hierarchy based on indentation and font size
   - **Standard extraction**: Expects title and page number on the same line (e.g., "Chapter 1 ... 15")
   - **OCR mode** (use `--ocr-mode`): Handles poorly scanned PDFs where titles and page numbers are on separate lines
3. **Validates** extracted page numbers against PDF page count (warns if out of range for excerpts)
4. **Matches** TOC entries to actual document locations using fuzzy text matching (excludes TOC pages from search)
5. **Creates** bookmarks that mirror the original TOC structure

### Method 2: Font-Based Detection (Fallback)

1. **Extracts** all text with font metadata (size, style, position)
2. **Analyzes** font size distribution to identify body text vs headings
3. **Classifies** headings by hierarchy level based on font size
4. **Generates** TOC structure with proper nesting

## Options Explained

### `--flat`
Creates a flat bookmark structure where all entries are at level 1 (no indentation/hierarchy).

**Without `--flat` (hierarchical):**
```
▪ Chapter 1
  • Section 1.1
    • Subsection 1.1.1
▪ Chapter 2
```

**With `--flat`:**
```
▪ Chapter 1
▪ Section 1.1
▪ Subsection 1.1.1
▪ Chapter 2
```

### `--no-existing-toc`
Forces the tool to use font-based detection instead of trying to extract from an existing TOC page. Useful if the PDF has a TOC page but you want to generate bookmarks based on actual headings.

### `--no-toc-bookmark`
Prevents adding a "Table of Contents" bookmark that points to the TOC page. Only applies when an existing TOC page is detected.

### `--ocr-mode`
Enables OCR-tolerant extraction for poorly scanned PDFs. Use this when:
- The PDF has low quality OCR text
- Titles and page numbers in the TOC are on separate lines
- Standard extraction produces garbage bookmarks or fails

**What it does:**
- Collects all text lines from TOC pages
- Separates titles from page numbers
- Combines fragmented section headers (e.g., "PART", "ONE" → "PART ONE")
- Validates page numbers against PDF page count
- Uses full-document fuzzy matching for out-of-range pages (common with excerpts)
- **Validates matches**: Only includes bookmarks with match score > 0.85 (skips TOC entries not found in document)

**When to use:**
- Scanned documents with poor OCR quality
- PDFs where TOC extraction produces individual words as bookmarks
- Excerpt PDFs where TOC references original document page numbers

**Important for excerpts:**  
If your PDF is an excerpt (e.g., 20 pages from a 200-page book), the tool will skip TOC entries for missing chapters. This is expected behavior and ensures bookmarks only point to content that actually exists in your PDF.

## Requirements

- Python 3.8+
- PyMuPDF (automatically installed)

## Development

### Using Jupyter Notebook

A Jupyter notebook version is included for exploration and customization:

```bash
uv run jupyter notebook pdf_toc_generator.ipynb
```

## Troubleshooting

### "No headings found"
- Try adjusting the font size detection threshold
- Use `--no-existing-toc` to force font-based detection
- Check if your PDF has selectable text (not scanned images)

### Poor quality bookmarks (individual words like "THE", "MINTO", etc.)
- Your PDF likely has poor OCR quality
- Use `--ocr-mode` flag to enable enhanced extraction
- In interactive mode, answer "yes" to "Is this a poorly scanned/OCR'd PDF?"

### TOC page numbers don't match document (e.g., TOC says page 45 but PDF only has 20 pages)
- This is normal for excerpt PDFs or scanned partial documents
- The tool automatically uses fuzzy matching to find headings in available pages
- Use `--ocr-mode` for better results with excerpts

### "File not found"
- Verify the PDF path is correct
- Use absolute paths or ensure you're in the correct directory
- Check file permissions

### Bookmarks not visible in PDF viewer
- Try a different PDF viewer (Adobe Acrobat, Preview, Chrome)
- Look for a bookmarks/outline panel (usually on the left sidebar)
- Some nested bookmarks may be collapsed - click the expand arrow

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

Built with [PyMuPDF](https://pymupdf.readthedocs.io/) for PDF processing.
