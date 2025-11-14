from pathlib import Path

lines = Path('requirements.txt').read_text(encoding='utf-16').splitlines()
print('total lines', len(lines))
for idx, line in enumerate(lines[:10], 1):
    print(idx, repr(line))
print('any blank lines?', any(not line.strip() for line in lines))
