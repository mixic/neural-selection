"""Run a minimal ORN-to-PN protocol in the authors' published NEURON model.

Requires the public ModelDBRepository/118662 checkout, compiled with nrnivmodl,
and the Python NEURON package. No files from that repository are copied here.
"""

import argparse
import csv
import json
import os
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--mode", choices=("current", "voltage"), default="current")
    parser.add_argument("--contact-ns", type=float, default=0.18)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--trace", type=Path,
                        help="write the somatic voltage or clamp current trace as CSV")
    args = parser.parse_args()

    os.chdir(args.model_dir.resolve())
    os.environ.pop("DISPLAY", None)
    from neuron import h

    h.load_file("stdrun.hoc")
    h.load_file("dm1_morph_2.hoc")
    cell = h.dm1_combined_12("assay_0", "dm1_combined_12", "")
    sections = list(cell.all)
    terminals = [sec for sec in cell.dendrite_group if len(sec.children()) == 0]
    for sec in sections:
        sec.Ra = 266.1
        sec.cm = 0.79
        for segment in sec:
            segment.LeakConductance.gmax = 1 / 20800
            segment.LeakConductance.e = -60

    rng = random.Random(args.seed)
    contacts = rng.sample(terminals, 25)
    stim = h.NetStim()
    stim.number = 1
    stim.start = 20
    stim.noise = 0
    synapses, connections = [], []
    for sec in contacts:
        synapse = h.Exp2Syn(sec(0.5))
        synapse.tau1 = 0.2
        synapse.tau2 = 1.1
        synapse.e = -10
        connection = h.NetCon(stim, synapse)
        connection.delay = 0
        connection.weight[0] = args.contact_ns / 1000  # nS to NEURON uS
        synapses.append(synapse)
        connections.append(connection)

    soma = cell.cable_0
    clamp = None
    clamp_current = None
    if args.mode == "voltage":
        clamp = h.SEClamp(soma(0.5))
        clamp.dur1 = 100
        clamp.amp1 = -65
        clamp.rs = 0.001
        clamp_current = h.Vector().record(clamp._ref_i)
    time = h.Vector().record(h._ref_t)
    voltage = h.Vector().record(soma(0.5)._ref_v)
    h.dt = 0.025
    h.finitialize(-65 if clamp else -60)
    h.continuerun(100)

    result = {"mode": args.mode, "contact_ns": args.contact_ns,
              "contacts": len(contacts), "terminal_branches": len(terminals)}
    if clamp is None:
        result["peak_epsp_mv"] = max(voltage) + 60
        trace = ((t, v + 60) for t, v in zip(time, voltage))
        trace_header = ("time_ms", "epsp_mv")
    else:
        baseline = [clamp_current[i] for i, t in enumerate(time) if 15 < t < 19]
        mean_baseline = sum(baseline) / len(baseline)
        result["peak_epsc_pa"] = -(min(clamp_current) - mean_baseline) * 1000
        trace = ((t, -(i - mean_baseline) * 1000)
                 for t, i in zip(time, clamp_current))
        trace_header = ("time_ms", "epsc_pa")
    if args.trace:
        with args.trace.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(trace_header)
            writer.writerows(trace)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
