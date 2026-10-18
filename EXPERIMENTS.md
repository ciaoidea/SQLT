# Previously recorded controlled experiment

This document summarizes a previously recorded synthetic numerical fixture.
It contains no operational memories, personal records or private conversations.
It is not a fresh run of the clean distribution. The original aggregate results
are reported here; a runnable benchmark reproducing this specific fixture is not
yet included in this source distribution.

## Task and mechanism

Events supplied explicit agent, patient and recipient bindings. An 8 by 16 tensor
was formed from role and participant identity codes. A role-axis readout computed
`prediction = event_tensor.T @ weights`. Eight coefficients were learned from
explicitly supplied outcomes using recursive least squares. Language generation
and the semantic encoder were not trained by this experiment.

## Recorded results

| Measure | Result |
| --- | ---: |
| Training examples | 16 |
| Held-out examples | 64 |
| Learned readout: participant selection | 64/64 |
| Nearest-example outcome copy | 0/64 |
| Shuffled-supervision participant selection | 64/64 |
| Learned vector mean squared error | 5.603146729926678e-8 |
| Shuffled-supervision vector mean squared error | 0.059994000766891986 |

The report also recorded successful memory retrieval from a fresh process.
This is evidence of persistence, not a test of resistance to catastrophic forgetting.

## Interpretation

Correct supervision yielded much more accurate vector reconstruction in this
fixture. Participant-selection accuracy alone did not distinguish the learned
readout from shuffled supervision. Copying a prior outcome failed on these
held-out cases; this is not a comparison with every possible retrieval baseline.
A handwritten recipient-selection rule also solves this simple task.

These results support a narrow learned numerical readout. They do not establish
general reasoning, broad superiority to retrieval, autonomous topological
reorganization, or retention under long sequences of new tasks. The integration
tests included with this distribution exercise storage, role direction,
negation, operator persistence, evidence removal and restart behavior; they are
separate from this historical 16/64 fixture.
