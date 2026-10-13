"""Review frozen DiffGui candidate pools with RDKit chemical alert catalogs."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, FilterCatalog

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'analysis/diffgui_literature_screen_dock10_261010'


def make_catalog(name: str) -> FilterCatalog.FilterCatalog:
    """Construct the unmodified RDKit catalog."""
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(getattr(FilterCatalog.FilterCatalogParams.FilterCatalogs, name))
    return FilterCatalog.FilterCatalog(params)


def grid(items: list[tuple], path: Path) -> None:
    """Draw readable two dimensional structures with identifiers and alerts."""
    molecules, legends = [], []
    for mol, row in items:
        copy = Chem.Mol(mol)
        AllChem.Compute2DCoords(copy)
        molecules.append(copy)
        alerts = row['brenk_alerts'] or 'none'
        legends.append(f"{row['id']} | dock {row['score']:.2f}\nBrenk: {alerts}\nPAINS: {row['pains_alerts'] or 'none'}")
    options = Draw.MolDrawOptions()
    options.legendFontSize = 18
    options.legendFraction = 0.25
    picture = Draw.MolsToGridImage(molecules, molsPerRow=3, subImgSize=(520, 360),
                                  legends=legends, drawOptions=options)
    picture.save(str(path))


def main() -> None:
    """Audit the saved final pools without changing the frozen selection."""
    catalogs = {name: make_catalog(name) for name in ('BRENK', 'PAINS')}
    rows, groups, stats = [], {}, {}
    for group, stem in [('Base', 'base'), ('Epoch80', 'epoch80')]:
        with (SOURCE / f'{stem}_with_r1r5_candidates.csv').open(encoding='utf-8-sig', newline='') as handle:
            metadata = list(csv.DictReader(handle))
        mols = list(Chem.SDMolSupplier(str(SOURCE / f'{stem}_with_r1r5_candidates.sdf'), removeHs=False))
        assert len(mols) == len(metadata)
        items = []
        for mol, original in zip(mols, metadata):
            assert mol is not None
            molecule = Chem.RemoveHs(mol)
            Chem.SanitizeMol(molecule)
            alerts = {name: sorted({match.GetDescription() for match in catalog.GetMatches(molecule)})
                      for name, catalog in catalogs.items()}
            row = dict(group=group, id=original['record_id'], score=float(original['score']),
                       smiles=Chem.MolToSmiles(molecule), valid=True,
                       formal_charge=Chem.GetFormalCharge(molecule),
                       radicals=sum(a.GetNumRadicalElectrons() for a in molecule.GetAtoms()),
                       fragments=len(Chem.GetMolFrags(molecule)),
                       mmff_parameters=bool(AllChem.MMFFHasAllMoleculeParams(Chem.AddHs(molecule))),
                       brenk_alerts='; '.join(alerts['BRENK']), pains_alerts='; '.join(alerts['PAINS']),
                       brenk_count=len(alerts['BRENK']), pains_count=len(alerts['PAINS']))
            rows.append(row)
            items.append((molecule, row))
        groups[group] = items
        stats[group] = dict(n=len(items), valid=sum(r['valid'] for _, r in items),
                            charged=sum(r['formal_charge'] != 0 for _, r in items),
                            radical=sum(r['radicals'] != 0 for _, r in items),
                            brenk_flagged=sum(r['brenk_count'] > 0 for _, r in items),
                            pains_flagged=sum(r['pains_count'] > 0 for _, r in items),
                            both_clear=sum(r['brenk_count'] == r['pains_count'] == 0 for _, r in items),
                            mmff_covered=sum(r['mmff_parameters'] for _, r in items),
                            brenk_categories=dict(Counter(a for _, r in items for a in r['brenk_alerts'].split('; ') if a)),
                            pains_categories=dict(Counter(a for _, r in items for a in r['pains_alerts'].split('; ') if a)))
        for page, start in enumerate(range(0, len(items), 12), 1):
            grid(items[start:start+12], OUT / f'{stem}_structures_{page:02d}.png')
    with (OUT / 'structure_audit.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / 'summary.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    epoch = groups['Epoch80']
    flagged = sorted([item for item in epoch if item[1]['brenk_count'] or item[1]['pains_count']],
                     key=lambda x: (-x[1]['brenk_count'], x[1]['score']))
    clear = sorted([item for item in epoch if not item[1]['brenk_count'] and not item[1]['pains_count']],
                   key=lambda x: x[1]['score'])
    if flagged:
        grid(flagged[:12], OUT / 'epoch80_flagged_examples.png')
    grid(clear[:9], OUT / 'epoch80_no_alert_examples.png')
    lookup = {row['id']: (mol, row) for mol, row in epoch}
    grid([lookup[key] for key in ['g6_224', 'g1_183', 'g6_175', 'g1_243', 'g1_137', 'g2_119']],
         OUT / 'manual_review_examples.png')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 17})
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = ['Base', 'Epoch80']
    passed = [stats[g]['both_clear'] for g in labels]
    alerted = [stats[g]['n'] - stats[g]['both_clear'] for g in labels]
    ax.bar(labels, passed, label='No Brenk / PAINS alert', color='#0874a7')
    ax.bar(labels, alerted, bottom=passed, label='Requires chemical review', color='#e5a052')
    for i, (p, a) in enumerate(zip(passed, alerted)):
        ax.text(i, p/2, str(p), ha='center', va='center', color='white', fontsize=22)
        if a:
            ax.text(i, p+a/2, str(a), ha='center', va='center', fontsize=20)
    ax.set_ylabel('Candidate count')
    ax.legend(loc='upper left', fontsize=13)
    ax.set_ylim(0, 160)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / 'chemical_alert_comparison.png', dpi=200)
    fig.savefig(OUT / 'chemical_alert_comparison.svg')
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
