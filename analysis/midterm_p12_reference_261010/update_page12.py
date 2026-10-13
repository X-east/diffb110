"""Update only the original deck's model-comparison page and its image."""
from __future__ import annotations

import csv
import json
import posixpath
from pathlib import Path
import shutil
from zipfile import ZipFile

from defusedxml.minidom import parseString


def main() -> None:
    """Patch one slide without round-tripping unrelated PowerPoint parts."""
    root=Path(__file__).resolve().parents[2]
    out=Path(__file__).parent
    source=root/'中期/谢东宸中期2.pptx'
    backup=root/'中期/备份/谢东宸中期2_修改第12页前_261010.pptx'
    backup.parent.mkdir(exist_ok=True)
    if backup.exists():raise FileExistsError(f'Existing backup: {backup}')
    shutil.copy2(source,backup)
    with ZipFile(source) as archive:parts={n:archive.read(n) for n in archive.namelist()}
    xml=parseString(parts['ppt/slides/slide12.xml'])
    changes={
        '五模型评价与强化学习底座选择':'模型横评',
        '比较已有模型在RE-CmeB口袋下的生成表现，选择强化学习探索的基础模型。':'结合已有模型评价结果，选择面向RE-CmeB的强化学习探索底座。',
        '−10.79':'−9.89',
        '在该批横评中，DiffGui的有效率、对接与QED表现较好。':'结合已有评价结果，DiffGui的有效率、对接与QED表现较好。',
        '500步历史横评；每模型3次生成重复、每次1000次尝试。有效率按全部尝试计算，图中白点为合并均值。':'对接为模型选型的评价参考值，图中白点为参考均值；有效率按生成尝试计算。',
        '对接单位kcal/mol，图表均使用成功记录的非正分子集；QED/SA为各重复合法样本均值的平均。正分原值保留于源表。':'DiffGui对接：训练早期评价（epoch1–20）；QED/SA按已有生成记录统计。对接单位kcal/mol。',
    }
    applied=[]
    for shape in xml.getElementsByTagName('p:sp'):
        nodes=shape.getElementsByTagName('a:t')
        value=''.join(n.firstChild.data if n.firstChild else '' for n in nodes)
        if value in changes:
            for index,node in enumerate(nodes):
                while node.firstChild:node.removeChild(node.firstChild)
                node.appendChild(xml.createTextNode(changes[value] if index==0 else ''))
            applied.append(value)
    assert len(applied)==len(changes),(applied,changes)
    parts['ppt/slides/slide12.xml']=xml.toxml(encoding='UTF-8')
    rels=parseString(parts['ppt/slides/_rels/slide12.xml.rels'])
    image_rels=[r for r in rels.getElementsByTagName('Relationship') if r.getAttribute('Type').endswith('/image')]
    assert len(image_rels)==1
    obsolete_image=posixpath.normpath(posixpath.join('ppt/slides',image_rels[0].getAttribute('Target')))
    image_rels[0].setAttribute('Target','../media/diffgui_page12_reference_261010.png')
    parts['ppt/slides/_rels/slide12.xml.rels']=rels.toxml(encoding='UTF-8')
    parts['ppt/media/diffgui_page12_reference_261010.png']=(out/'model_comparison.png').read_bytes()
    parts.pop(obsolete_image)
    notes_path='ppt/notesSlides/notesSlide12.xml'
    if notes_path in parts:
        notes=parseString(parts[notes_path])
        for shape in notes.getElementsByTagName('p:sp'):
            placeholders=shape.getElementsByTagName('p:ph')
            if not any(ph.getAttribute('type')=='body' for ph in placeholders):continue
            nodes=shape.getElementsByTagName('a:t')
            if nodes:
                for index,node in enumerate(nodes):
                    while node.firstChild:node.removeChild(node.firstChild)
                    node.appendChild(notes.createTextNode('模型横评页用于底座选型。DiffGui对接参考值−9.89来自主训练epoch1–20逐轮对接均值的平均；密度图使用对应成功非正样本。其他模型数据以及有效率、QED、标准SA来自已有横评记录。本页不将−9.89记作未调优底座500步三重复均值，也不直接与epoch80独立生成均值相减宣称调优增益。原始记录见analysis/midterm_p12_reference_261010/figure_source.json。' if index==0 else ''))
        parts[notes_path]=notes.toxml(encoding='UTF-8')
    temporary=out/'updated.pptx'
    with ZipFile(temporary,'w') as archive:
        for name,data in parts.items():archive.writestr(name,data)
    with ZipFile(backup) as original,ZipFile(temporary) as revised:
        changed=[n for n in original.namelist() if n in revised.namelist() and original.read(n)!=revised.read(n)]
        expected={'ppt/slides/slide12.xml','ppt/slides/_rels/slide12.xml.rels',notes_path}
        assert set(changed)<=expected,changed
        for i in range(1,26):
            if i!=12:assert original.read(f'ppt/slides/slide{i}.xml')==revised.read(f'ppt/slides/slide{i}.xml')
    shutil.copyfile(temporary,source)
    (out/'edit_manifest.json').write_text(json.dumps({'source':str(source),'backup':str(backup),'slide':12,'changed_parts':changed,'removed_unused_image':obsolete_image,'replacements':changes},ensure_ascii=False,indent=2))
    print('Updated only page 12:',source)
    print('Backup:',backup)


if __name__=='__main__':main()
