#!/usr/bin/env python3
"""GPU integration: all execution modes, placebo parity and ablation resume."""
import argparse
import csv
import hashlib
from pathlib import Path
import subprocess
import tempfile


def run(exe, *args, success=True):
    p=subprocess.run([str(exe), *map(str,args)], capture_output=True, text=True)
    if (p.returncode == 0) != success: raise AssertionError(p.stdout+p.stderr)


def rows(path):
    with Path(path).open() as f: return list(csv.DictReader(f))


def main():
    p=argparse.ArgumentParser();p.add_argument('executable',type=Path);args=p.parse_args()
    exe=args.executable.resolve()
    with tempfile.TemporaryDirectory(prefix='plasticity-') as temp:
        root=Path(temp)
        for mode,flags in [('global',[]),('legacy',['--legacy-neural']),
                           ('block',['--block-brains']),('event',['--event-synapses'])]:
            folder=root/mode;folder.mkdir()
            source=folder/'source.gz';base=folder/'baseline.csv'
            run(exe,'--population',8,'--seed',2,'--steps',50,'--output',base,
                '--checkpoint',source,*flags)
            original=hashlib.sha256(source.read_bytes()).hexdigest()
            prefix=folder/'frozen.csv';branch=folder/'branch.gz'
            opts=['--resume',source,'--output',prefix,'--checkpoint',branch,
                  '--plasticity-ablation','--ablation-fraction',1,'--ablation-seed',42]
            run(exe,*opts,'--continuation-steps',100)
            initial=rows(str(base)+'.synapses.csv')
            final=rows(str(prefix)+'.synapses.csv')
            assert len(initial)==len(final) and initial
            for before,after in zip(initial,final):
                # Genotype, learned efficacy and structural plasticity all remain fixed.
                assert before==after, (mode,before,after)
            cells_before=rows(str(base)+'.neurons.csv');cells_after=rows(str(prefix)+'.neurons.csv')
            assert any(int(b['total_spikes'])>int(a['total_spikes'])
                       for a,b in zip(cells_before,cells_after)), 'Frozen neurons did not fire'
            for before,after in zip(cells_before,cells_after):
                assert before['ampa_scale']==after['ampa_scale']
                for key in before:
                    if key not in ('total_spikes','receptor_state_ns'):
                        assert before[key]==after[key], (mode,key)
            assert hashlib.sha256(source.read_bytes()).hexdigest()==original
            # Zero treated animals: intervention must have no effect on the world.
            control=folder/'control.csv';placebo=folder/'placebo.csv'
            run(exe,'--resume',source,'--output',control,'--checkpoint',folder/'control.gz',
                '--continuation-steps',100)
            run(exe,'--resume',source,'--output',placebo,'--checkpoint',folder/'placebo.gz',
                '--plasticity-ablation','--ablation-fraction',0,'--continuation-steps',100)
            for suffix in ('.synapses.csv','.neurons.csv','.png'):
                assert Path(str(control)+suffix).read_bytes()==Path(str(placebo)+suffix).read_bytes(),(mode,suffix)
            assert Path(str(control)+'.synapses.csv').read_bytes()!=Path(str(base)+'.synapses.csv').read_bytes(), 'Control did not learn'
            # Compare continuous vs checkpointed intervention including observer records.
            direct=folder/'direct.csv'
            run(exe,'--resume',source,'--output',direct,'--checkpoint',folder/'direct.gz',
                '--plasticity-ablation','--ablation-fraction',1,'--ablation-seed',42,
                '--continuation-steps',150)
            run(exe,'--resume',branch,'--continuation-steps',50)
            for suffix in ('.plasticity.csv','.synapses.csv','.neurons.csv','.png'):
                assert Path(str(prefix)+suffix).read_bytes()==Path(str(direct)+suffix).read_bytes(),(mode,suffix)
            run(exe,'--resume',source,'--output',folder/'invalid.csv','--plasticity-ablation',
                '--continuation-steps',10,success=False) # source checkpoint protection
            print(f'{mode}: freeze, ongoing firing, placebo parity and checkpoint resume passed',flush=True)
        # Complete the multi-trial runner and its persisted summary.
        subprocess.run(['python3','tools/plasticity_continuation.py','--executable',str(exe),
                        '--checkpoint',str(source),'--output',str(root/'trials'),
                        '--trials','3','--steps','20'],check=True)
        import json
        report=json.loads((root/'trials/results.json').read_text())
        assert report['completed_trials']==3
        assert all(t['groups']['control']['n']==4 and t['groups']['frozen']['n']==4 for t in report['trials'])
    print('Repeated-trial runner passed')


if __name__=='__main__': main()
