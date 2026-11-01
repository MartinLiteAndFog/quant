"""Offline CSV importer; inputs are never modified and no network is used."""
import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
from . import Provenance, normalize, synthetic_week


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path)
    parser.add_argument('--provenance', type=Path, help='JSON Provenance fields required with CSV')
    parser.add_argument('--start')
    parser.add_argument('--end')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.csv:
        if not args.provenance:
            parser.error('--provenance required with --csv')
        provenance = Provenance(**json.loads(args.provenance.read_text()))
        with args.csv.open(newline='') as source:
            dataset = normalize(csv.DictReader(source), provenance, start=args.start, end=args.end)
    else:
        if args.provenance or args.start or args.end:
            parser.error('range/provenance options require --csv')
        dataset = synthetic_week()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'quality.json').write_text(json.dumps(dataset.report, indent=2)+'\n')
    with (args.output/'bars.jsonl').open('w') as target:
        for bar in dataset.bars:
            target.write(json.dumps(asdict(bar), default=str, sort_keys=True)+'\n')
    print(json.dumps(dataset.report, sort_keys=True))


if __name__ == '__main__':
    main()
