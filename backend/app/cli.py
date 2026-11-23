"""CLI dataset import - no arbitrary-server-file-path endpoint."""
import argparse
from pathlib import Path
from .db import initialize
from .knowledge.ingest import import_qasper_file


def main():
    p=argparse.ArgumentParser(description='Import QASPER JSON without exposing gold answers to retrieval')
    p.add_argument('json',type=Path)
    p.add_argument('--split',choices=['train','dev','validation','test'],default='dev')
    p.add_argument('--limit',type=int,default=None)
    args=p.parse_args()
    if not args.json.is_file():p.error('Input must be an existing QASPER JSON file')
    initialize()
    print(import_qasper_file(args.json,args.split,args.limit))


if __name__=='__main__':main()
