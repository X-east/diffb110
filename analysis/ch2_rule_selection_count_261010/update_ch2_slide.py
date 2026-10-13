"""Replace only the first-slide violin figure and its width caption in a review copy."""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

from defusedxml.minidom import parseString
from pptx import Presentation

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REL_NS = 'http://schemas.openxmlformats.org/package/2006/relationships'


def main() -> None:
    """Preserve four other slides byte for byte and leave both source decks intact."""
    source = ROOT / '中期/第二部分_结构偏好优化结果_五页_v2.pptx'
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / '中期/第二部分_结构偏好优化结果_五页_v3.pptx'
    assert not output.exists(), output
    presentation = Presentation(source)
    picture = next(shape for shape in presentation.slides[0].shapes if shape.shape_type == 13)
    relationship_id = picture._element.blipFill.blip.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
    caption = '蓝色小提琴最大宽度按各模型筛选保留比例缩放；横轴数字为筛选前→后数量。白点为对接均值。'
    with zipfile.ZipFile(source) as reader:
        payload = {name: reader.read(name) for name in reader.namelist()}
    rel_path = 'ppt/slides/_rels/slide1.xml.rels'
    rels = parseString(payload[rel_path])
    new_media = 'count_scaled_rule_violin_261010.png'
    old_media = None
    for node in rels.getElementsByTagNameNS(REL_NS, 'Relationship'):
        if node.getAttribute('Id') == relationship_id:
            old_media = node.getAttribute('Target').split('/')[-1]
            node.setAttribute('Target', '../media/' + new_media)
    payload[rel_path] = rels.toxml(encoding='UTF-8')
    slide = parseString(payload['ppt/slides/slide1.xml'])
    changed_caption = False
    for shape in slide.getElementsByTagName('p:sp'):
        nodes = shape.getElementsByTagName('a:t')
        old = ''.join(node.firstChild.data if node.firstChild else '' for node in nodes)
        if old.startswith('两组对接均使用历史成功记录'):
            for node in nodes:
                while node.firstChild:
                    node.removeChild(node.firstChild)
            nodes[0].appendChild(slide.createTextNode(caption))
            changed_caption = True
    assert changed_caption
    payload['ppt/slides/slide1.xml'] = slide.toxml(encoding='UTF-8')
    payload['ppt/media/' + new_media] = (HERE / 'docking_violin_count_scaled.png').read_bytes()
    if old_media and not any(old_media.encode() in data for name, data in payload.items() if name.endswith('.rels')):
        payload.pop('ppt/media/' + old_media)
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as writer:
        for name, data in payload.items():
            writer.writestr(name, data)
    with zipfile.ZipFile(source) as a, zipfile.ZipFile(output) as b:
        for index in range(2, 6):
            name = f'ppt/slides/slide{index}.xml'
            assert a.read(name) == b.read(name)
    (HERE / 'ppt_edit_manifest.json').write_text(json.dumps(dict(source=str(source), output=str(output),
                       changed_slide=1, unchanged_slides=[2,3,4,5], new_media=new_media), ensure_ascii=False, indent=2))
    print(output)


if __name__ == '__main__':
    main()
