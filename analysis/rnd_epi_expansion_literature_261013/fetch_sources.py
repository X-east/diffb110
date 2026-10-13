"""Archive open article XML and metadata for the EPI expansion literature audit."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
from pathlib import Path
import xml.etree.ElementTree as ET

import requests


LOGGER = logging.getLogger(__name__)
API_ROOT = "https://www.ebi.ac.uk/europepmc/webservices/rest"
QUERIES = {
    "nora_pharmacophore_2020": "PMC7582826",
    "epi_screen_2020": "PMC7153915",
    "epi_aggregation_2023": "PMC10296211",
    "consensus_2017": "PMC5705545",
    "nora_bioisostere_2021": "PMC8415528",
    "swisssimilarity_2022": "DOI:10.3390/ijms23020811",
    "acrb_pharmacophore_2024": "DOI:10.1039/D3MD00483J",
    "mexy_berberine_2023": "DOI:10.1038/s44259-023-00013-4",
    "nora_pharmacophore_2017": "EXT_ID:28117588 AND SRC:MED",
    "nora_fields_2011": "EXT_ID:21141825 AND SRC:MED",
    "zipa_shape_2005": "EXT_ID:15743191 AND SRC:MED",
    "pharmit_2016": "DOI:10.1093/nar/gkw287",
}


def fetch_source(item: tuple[str, str], output_dir: Path) -> dict[str, object]:
    """Fetch primary article metadata and openly available full text."""
    source_id, query = item
    result: dict[str, object] = {"source_id": source_id, "query": query}
    try:
        response = requests.get(
            f"{API_ROOT}/search",
            params={"query": query, "format": "json", "resultType": "core"},
            timeout=25,
        )
        response.raise_for_status()
        records = response.json()["resultList"]["result"]
        if len(records) != 1:
            raise ValueError(f"Expected one article, obtained {len(records)}")
        metadata = records[0]
        result.update(metadata)
        pmcid = metadata.get("pmcid")
        if pmcid and metadata.get("isOpenAccess") == "Y":
            xml_url = f"{API_ROOT}/{pmcid}/fullTextXML"
            xml_response = requests.get(xml_url, timeout=30)
            xml_response.raise_for_status()
            root = ET.fromstring(xml_response.content)
            (output_dir / f"{source_id}.xml").write_bytes(xml_response.content)
            body = root.find("body")
            if body is None:
                raise ValueError("Article XML does not contain a body")
            rows = [
                " ".join(" ".join(node.itertext()).split())
                for node in body.iter()
                if node.tag in {"title", "p", "table-wrap", "fig"}
            ]
            (output_dir / f"{source_id}.txt").write_text(
                "\n\n".join(rows) + "\n", encoding="utf-8"
            )
            result["archived_xml_url"] = xml_url
            result["verification"] = "full_text_xml"
        else:
            result["verification"] = "metadata_and_abstract"
        LOGGER.info("%s: %s", source_id, result["verification"])
    except (requests.RequestException, ValueError, KeyError, ET.ParseError) as exc:
        result["error"] = str(exc)
        LOGGER.warning("%s: %s", source_id, exc)
    return result


def main() -> None:
    """Archive sources and preserve successes and failures in a JSON manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "sources")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(args.output_dir / "fetch.log")],
    )
    manifest_path = args.output_dir / "source_manifest.json"
    cached = {}
    if manifest_path.exists():
        cached = {row["source_id"]: row for row in json.loads(manifest_path.read_text())
                  if not row.get("error")}
    pending = [(source_id, query) for source_id, query in QUERIES.items()
               if source_id not in cached]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        fetched = list(pool.map(lambda item: fetch_source(item, args.output_dir), pending))
    cached.update({row["source_id"]: row for row in fetched})
    results = [cached[source_id] for source_id in QUERIES]
    manifest_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for result in results:
        print(json.dumps({key: result.get(key) for key in
                         ["source_id", "title", "doi", "pmcid", "verification", "error"]},
                         ensure_ascii=False))


if __name__ == "__main__":
    main()
