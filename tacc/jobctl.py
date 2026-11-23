#!/usr/bin/env python3
"""Safe Slurm submit/status/cancel wrapper; no cluster credentials included."""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def submit(args):
    file=Path(args.script).resolve()
    cfg=Path(args.config).resolve()
    if not file.is_file() or not cfg.is_file():
        raise ValueError('Job script and private configuration file must both exist')
    if not args.partition or not args.account:
        raise ValueError('Explicit --partition and --account are required; consult course allocation')
    if args.gpus < 1 or args.gpus>8:
        raise ValueError('GPU count must be within 1..8')
    if args.gpu_syntax=='gres':
        gpu_request=f'--gres=gpu:{args.gpus}'
    else:
        gpu_request=f'--gpus={args.gpus}'
    command=['sbatch','--parsable','--partition',args.partition,'--account',args.account,
             gpu_request,'--export=ALL,TACC_CONFIG_FILE='+str(cfg),str(file)]
    if args.dry_run:
        print(' '.join(command));return 0
    proc=subprocess.run(command,check=True,capture_output=True,text=True)
    job=proc.stdout.strip().split(';',1)[0]
    if not re.fullmatch(r'\d+',job):
        raise RuntimeError(f'Unexpected sbatch response: {proc.stdout[:120]}')
    print(f'Submitted job {job}')
    print('Status: python3 tacc/jobctl.py status '+job)
    return 0


def status(args):
    if not re.fullmatch(r'\d+',args.job):
        raise ValueError('Job ID must be numeric')
    cmd=['squeue','--noheader','--jobs',args.job,'--format=%i|%T|%N|%M|%l']
    proc=subprocess.run(cmd,capture_output=True,text=True,check=True)
    if not proc.stdout.strip():
        print('Job not in active queue; it may have finished or failed.')
        print('Check accounting: sacct -j '+args.job+' --format=JobID,State,Elapsed,ExitCode')
        return 0
    print('JobID|State|Node|Elapsed|TimeLimit')
    print(proc.stdout.strip())
    return 0


def cancel(args):
    if not re.fullmatch(r'\d+',args.job):raise ValueError('Job ID must be numeric')
    if not args.yes:
        raise ValueError('Cancellation requires explicit --yes')
    subprocess.run(['scancel',args.job],check=True)
    print('Cancellation requested for job '+args.job)
    return 0


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sp=p.add_subparsers(dest='action',required=True)
    sub=sp.add_parser('submit')
    sub.add_argument('--script',default='tacc/job_template.sbatch')
    sub.add_argument('--config',required=True)
    sub.add_argument('--partition',required=True)
    sub.add_argument('--account',required=True)
    sub.add_argument('--gpus',type=int,default=2)
    sub.add_argument('--gpu-syntax',choices=['gres','gpus'],default='gres')
    sub.add_argument('--dry-run',action='store_true')
    sub=sp.add_parser('status');sub.add_argument('job')
    sub=sp.add_parser('cancel');sub.add_argument('job');sub.add_argument('--yes',action='store_true')
    args=p.parse_args(argv)
    try:return {'submit':submit,'status':status,'cancel':cancel}[args.action](args)
    except (ValueError,subprocess.CalledProcessError,RuntimeError) as exc:
        print('TACC JOBCTL ERROR:',exc,file=sys.stderr)
        return 2

if __name__=='__main__':raise SystemExit(main())
