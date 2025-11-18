import fitz
import sys

doc = fitz.open(sys.argv[1])
toc = doc.get_toc()
for level, title, page in toc:
    indent = '  ' * (level - 1)
    print(f'L{level}: {indent}{title} → page {page}')
doc.close()
