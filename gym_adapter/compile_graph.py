"""Eliminate graph components with no causal path to observable motor outputs.

The genome remains unchanged. Original neuron/synapse indices are retained for
stochastic reset streams. Every active edge, including inhibition/modulation,
participates in reachability; local plasticity cannot activate an absent edge.
"""
from .adapter import Genome


def compile_genome(genome):
    incoming=[[] for _ in genome.neurons]
    for e in genome.synapses:
        a,b=int(e[0]),int(e[1])
        if e[3] and genome.neurons[a][2] and genome.neurons[b][2]:
            incoming[b].append(a)
    live=set(genome.motors);pending=list(live)
    while pending:
        for a in incoming[pending.pop()]:
            if a not in live:live.add(a);pending.append(a)
    indices=sorted(live);remap={old:new for new,old in enumerate(indices)}
    neurons=[genome.neurons[i][:] for i in indices];edges=[];edge_indices=[]
    for j,e in enumerate(genome.synapses):
        a,b=int(e[0]),int(e[1])
        if e[3] and a in live and b in live and genome.neurons[a][2] and genome.neurons[b][2]:
            edge=e[:];edge[0]=remap[a];edge[1]=remap[b];edges.append(edge);edge_indices.append(j)
    result=Genome(neurons,edges,[remap[m] for m in genome.motors],genome.channels)
    result._original_neurons=indices;result._original_synapses=edge_indices
    return result
