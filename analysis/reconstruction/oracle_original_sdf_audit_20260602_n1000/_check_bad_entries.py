"""Check why specific SDF entries fail to parse."""
import os

sdf_path = os.path.join(os.path.dirname(__file__), 'original_vs_repaired_oracle_pairs.sdf')
with open(sdf_path) as f:
    content = f.read()

entries = content.split('$$$$\n')

# Entry 16 is the first parse failure (from RDKit)
for idx in [16, 42, 48, 52, 58, 88]:
    if idx >= len(entries):
        continue
    e = entries[idx].strip()
    lines = e.split('\n')

    # Show the first line (name)
    print(f'=== Entry {idx}: {lines[0][:80]}')

    # Find V2000
    for j, l in enumerate(lines):
        if 'V2000' in l:
            parts = l.strip().split()
            try:
                na, nb = int(parts[0]), int(parts[1])
            except (ValueError, IndexError) as e:
                print(f'  BAD V2000 line: [{l}] error={e}')
                continue
            print(f'  V2000 line[{j}]: [{l}]')
            print(f'  atoms={na}, bonds={nb}')

            # Show atom block (first 3)
            for k in range(min(3, na)):
                al = lines[j+1+k].rstrip()
                print(f'  atom[{k}]: len={len(al)} [{al}]')

            # Show bond block (first 3)
            bstart = j + 1 + na
            bend = bstart + nb
            for k in range(min(3, nb)):
                bl = lines[bstart+k].rstrip()
                print(f'  bond[{k}]: len={len(bl)} [{bl}]')

            # Show M END
            if bend < len(lines):
                print(f'  MEND[{bend}]: [{lines[bend].rstrip()}]')
            else:
                print(f'  NO M END: need line {bend}, have {len(lines)} lines')
            break
    print()
