"""Set separate Latin and East Asian typefaces throughout the OOXML package."""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

from defusedxml.minidom import parseString

DRAWING_NS = 'http://schemas.openxmlformats.org/drawingml/2006/main'


def main() -> None:
    """Patch font declarations while preserving namespace prefixes."""
    source, target = map(Path, sys.argv[1:3])
    assert not target.exists(), target
    nodes, changed = 0, []
    with zipfile.ZipFile(source) as reader, zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as writer:
        for item in reader.infolist():
            content = reader.read(item.filename)
            if item.filename.startswith('ppt/') and item.filename.endswith('.xml'):
                document = parseString(content)
                found = False
                for tag in ['rPr', 'defRPr', 'endParaRPr', 'majorFont', 'minorFont']:
                    for node in list(document.getElementsByTagNameNS(DRAWING_NS, tag)):
                        for child_tag, font in [('latin', 'Arial'), ('ea', 'Microsoft YaHei'), ('cs', 'Arial')]:
                            children = [child for child in node.childNodes if child.nodeType == child.ELEMENT_NODE
                                        and child.namespaceURI == DRAWING_NS and child.localName == child_tag]
                            if children:
                                child = children[0]
                            else:
                                child = document.createElementNS(DRAWING_NS, f'a:{child_tag}')
                                # Typeface declarations precede hyperlinks, if any.
                                anchor = next((c for c in node.childNodes if c.nodeType == c.ELEMENT_NODE
                                               and c.localName in ('hlinkClick', 'hlinkMouseOver', 'extLst')), None)
                                if anchor:
                                    node.insertBefore(child, anchor)
                                else:
                                    node.appendChild(child)
                            child.setAttribute('typeface', font)
                        nodes += 1
                        found = True
                if found:
                    changed.append(item.filename)
                    content = document.toxml(encoding='UTF-8')
            writer.writestr(item, content)
    (source.parent / 'font_audit.json').write_text(json.dumps({'font_nodes_patched': nodes,
                  'parts': changed, 'latin': 'Arial', 'east_asian': 'Microsoft YaHei'}, indent=2))
    print(f'Explicit typefaces patched in {nodes} font nodes.')


if __name__ == '__main__':
    main()
