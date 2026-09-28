from pathlib import Path
import difflib
import hashlib
import json
import re
import subprocess
import sys

ROOT = Path('/home/likefallwind/code/llm-knowledge-graph')
RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from kg.sources import chunk_text
import kg.sources as sources

pdf = ROOT / 'data/docs/sutton-barto.pdf'
raw = subprocess.check_output(['pdftotext', '-layout', str(pdf), '-'], text=True)
(RUN / 'original-extraction.txt').write_text(raw)
pages = raw.split('\f')
toc = '\n'.join(pages[6:12])
entries = []
pending = ''
for line in toc.splitlines():
    match = re.match(r'^\s*(\d+(?:\.\d+)*)\s+(.+)', line)
    if match:
        pending = match.group(1) + ' ' + match.group(2)
    elif pending and line.strip():
        pending += ' ' + line.strip()
    else:
        continue
    end = re.search(r'\s(\d+)\s*$', pending)
    if end:
        number, title = pending[:end.start()].split(' ', 1)
        title = re.sub(r'(?:\s*\.){2,}\s*$', '', title).strip()
        entries.append(dict(number=number, title=title, page=int(end.group(1))))
        pending = ''
assert len(entries) == 183
chapters = [e for e in entries if '.' not in e['number']]
assert [int(e['number']) for e in chapters] == list(range(1, 18))
start = next(i for i, page in enumerate(pages) if page.startswith('Chapter 1\n'))
offset = start
assert offset == 22
for entry in chapters:
    assert pages[entry['page'] + offset - 1].startswith('Chapter ' + entry['number'] + '\n')
extras = []
for line in toc.splitlines():
    match = re.match(r'^\s*(I{1,3})\s+(.+?)\s{2,}(\d+)\s*$', line)
    if match:
        extras.append(dict(number=match.group(1), title=match.group(2), page=int(match.group(3)), kind='part'))
    match = re.match(r'^(References|Index)\s+(\d+)\s*$', line)
    if match:
        extras.append(dict(number='', title=match.group(1), page=int(match.group(2)), kind='back_matter'))
assert len(extras) == 5
by_page = {}
for entry in entries + extras:
    by_page.setdefault(entry['page'] + offset, []).append(entry)

clean = [''] * start
headers = []
headings = []
norm = lambda value: re.sub(r'\W', '', value).lower()
for i, page in enumerate(pages[start:], start):
    lines = page.splitlines()
    first = next((j for j, line in enumerate(lines) if line.strip()), None)
    printed_page = i + 1 - offset
    if first is not None:
        line = lines[first]
        if re.fullmatch(r'\s*' + str(printed_page) + r'\s{2,}\S.*', line) or re.fullmatch(r'.*\S\s{2,}' + str(printed_page) + r'\s*', line):
            headers.append(dict(pdf_page=i+1, text=line))
            lines[first] = ''
    replacements = {}
    for entry in by_page.get(i + 1, []):
        kind = entry.get('kind', 'section' if '.' in entry['number'] else 'chapter')
        if kind == 'section':
            pattern = r'^\s*' + re.escape(entry['number']) + r'\s+'
        elif kind == 'chapter':
            pattern = r'^Chapter ' + entry['number'] + r'\s*$'
        elif kind == 'part':
            pattern = r'^Part ' + entry['number'] + r':(?:\s.*)?$'
        else:
            pattern = '^' + entry['title'] + '$'
        hits = [j for j, line in enumerate(lines) if re.match(pattern, line)]
        assert len(hits) == 1, (entry, hits, i+1)
        j = hits[0]
        consumed = 1
        if kind == 'section':
            body_title = re.sub(pattern, '', lines[j])
            for count in range(1, 4):
                combined = ' '.join([body_title] + lines[j+1:j+count])
                if norm(combined) == norm(entry['title']):
                    consumed = count
                    break
            score = difflib.SequenceMatcher(None, norm(body_title), norm(entry['title'])).ratio()
            assert score > 0.45, (entry, body_title)
            level = entry['number'].count('.') + (1 if entry['number'].startswith('1.') else 2)
            title = entry['number'] + ' ' + entry['title']
        elif kind == 'chapter':
            level = 1 if entry['number'] == '1' else 2
            title = 'Chapter ' + entry['number'] + ': ' + entry['title']
        else:
            level = 1
            title = ('Part ' + entry['number'] + ': ' if kind == 'part' else '') + entry['title']
        heading = '#' * level + ' ' + title
        replacements[j] = (consumed, '\n' + heading + '\n')
        headings.append(dict(pdf_page=i+1, printed_page=printed_page, level=level, title=title,
            marker=heading, original_lines=lines[j:j+consumed], kind=kind))
    rendered = []
    j = 0
    while j < len(lines):
        if j in replacements:
            consumed, replacement = replacements[j]
            rendered.append(replacement)
            j += consumed
        else:
            rendered.append(lines[j])
            j += 1
    clean.append('\n'.join(rendered))
text = '\f'.join(clean)
assert text.count('\f') == raw.count('\f')
assert len(headings) == 188
heading_map = {e['marker']: (e['level'], e['title']) for e in headings}
chunks = chunk_text(text, headings=heading_map)
assert all(len({p.section_path for p in c.passages}) == 1 for c in chunks)
assert 'page 23,' in chunks[0].passages[0].location
assert {p.section_path[-1] for c in chunks for p in c.passages} == {e['title'] for e in headings}
for c in chunks:
    for p in c.passages:
        assert text[p.start:p.end] == p.text
(RUN / 'book.txt').write_text(text)
(RUN / 'toc.json').write_text(json.dumps(entries + extras, ensure_ascii=False, indent=2))
(RUN / 'headings.json').write_text(json.dumps(headings, ensure_ascii=False, indent=2))
(RUN / 'removed-headers.json').write_text(json.dumps(headers, ensure_ascii=False, indent=2))
(RUN / 'chunks.json').write_text(json.dumps([dict(index=c.index, sha256=c.content_hash,
    section_path=c.section_path, location=c.location, chars=len(c.text)) for c in chunks], ensure_ascii=False, indent=2))
report = dict(total_chunks=len(chunks), chapters=17, numbered_sections=166, parts=3,
    back_matter_sections=2, total_sections=len(headings), removed_headers=len(headers),
    original_pdf_page_count=len(pages)-1, first_evidence_page=23,
    original_text_sha256=hashlib.sha256(raw.encode()).hexdigest(),
    prepared_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
    pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
    checks=dict(all_toc_headings_matched=True, no_cross_section_chunks=True,
                exact_passage_slices=True, physical_page_positions_preserved=True),
    notes=['Headings follow the original TOC and are matched on their printed page.',
           'Body and back matter retained; front matter is excluded with page delimiters preserved.',
           'No generated formula repairs; original PDF and extracted text remain the audit reference.',
           'Section 5.4 title is recovered from the TOC because plot text interleaves its last word.',
           'Old pilot chunk indices are not reusable after correcting section boundaries.'])
(RUN / 'preprocessing-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(json.dumps(report, ensure_ascii=False, indent=2))
