# Literature-search queries (Scopus and Web of Science)

Boolean queries used for the literature search reported in the manuscript
"Domain- and Biology-Aware Screening of Calcium Phosphate Bone-Scaffold
Candidates with Graph Neural Networks and Density Functional Theory
Validation" (IEEE Access).

**Search date:** 22 November 2026.
**Databases:** Scopus (scopus.com) and Web of Science (webofscience.com),
run directly by the first author from the Universitas Airlangga campus
network.

## How to run them

- **Scopus:** open the basic search, set the field to "Article title,
  Abstract, Keywords", and paste the query WITHOUT the `TITLE-ABS-KEY(...)`
  wrapper.
- **Web of Science:** open Advanced Search, go to the Query Builder, and
  paste the full query (WITH the `TS=(...)` wrapper) into the Query Preview
  box, then Search. (The Smart Search box and the fielded Topic box reject
  the nested Boolean; the Query Preview route is the working one.)

## String A — the triple combination (novelty claim a)

GNN + active learning + DFT for calcium-phosphate scaffolds.

- Scopus:
  `TITLE-ABS-KEY(("graph neural network" OR "crystal graph" OR CGCNN OR MEGNet OR ALIGNN) AND ("active learning") AND ("density functional theory" OR DFT) AND ("calcium phosphate" OR hydroxyapatite OR "bone scaffold"))`
- WoS:
  `TS=(("graph neural network" OR "crystal graph" OR CGCNN OR MEGNet OR ALIGNN) AND "active learning" AND ("density functional theory" OR DFT) AND ("calcium phosphate" OR hydroxyapatite OR "bone scaffold"))`
- Result: **Scopus 0 documents; WoS 0 documents.**

## String B — GNN + calcium phosphate (broader)

- Scopus:
  `TITLE-ABS-KEY(("graph neural network" OR "crystal graph" OR CGCNN) AND ("calcium phosphate" OR hydroxyapatite OR "beta-tricalcium phosphate" OR "bone scaffold"))`
- WoS:
  `TS=(("graph neural network" OR "crystal graph" OR CGCNN) AND ("calcium phosphate" OR hydroxyapatite OR "beta-tricalcium phosphate" OR "bone scaffold"))`
- Result: **Scopus 0 documents; WoS 1 document** — "Machine learning on
  properties of multiscale multisource hydroxyapatite nanoparticles datasets
  with different morphologies and sizes" (Liu, Shi & Ma, npj Computational
  Materials 7(1), 2021): ML for hydroxyapatite nanoparticle properties, not
  scaffold screening, not GNN + active learning + DFT. Not a counterexample.

## String C — biocompatibility gate (novelty claim b)

- Scopus:
  `TITLE-ABS-KEY(("biocompatibility" OR "biocompatible" OR toxicity OR cytotoxicity) AND (gate OR filter) AND (screening) AND ("calcium phosphate" OR hydroxyapatite))`
- WoS:
  `TS=(("biocompatibility" OR "biocompatible" OR toxicity OR cytotoxicity) AND (gate OR filter) AND (screening) AND ("calcium phosphate" OR hydroxyapatite))`
- Result: **Scopus 2 documents; WoS 0 documents.** The two Scopus hits are
  (1) an ISCEMP 2013 conference proceedings (not a research paper) and
  (2) "Potential essentiality of lead" (Schwarz, 1975, about lead, not
  relevant). Neither is a counterexample.

## String D — ML screening of calcium phosphate (broadest)

- Scopus:
  `TITLE-ABS-KEY(("machine learning" OR "artificial intelligence") AND (screening) AND ("calcium phosphate" OR hydroxyapatite OR "bone scaffold"))`
- WoS:
  `TS=(("machine learning" OR "artificial intelligence") AND (screening) AND ("calcium phosphate" OR hydroxyapatite OR "bone scaffold"))`
- Result: **Scopus 24 documents; WoS 16 documents.** General ML-for-calcium-
  phosphate literature (reviews, editorials, ML for properties, ML for pore
  geometry, ML for ink viscosity, etc.). Because Strings A and B returned
  zero, none of these uses a graph neural network, so none is a
  counterexample to either novelty claim. The closest hits are Pradana et
  al. (2025, hydroxyapatite crystallinity by ML), the high-throughput
  screening + ML study of calcium-phosphate scaffold structures
  (Biomaterials, 2025), and the AI-selected oligopeptide-coated biphasic
  calcium phosphate scaffold study (Advanced Healthcare Materials, 2026) —
  all already cited and distinguished in the manuscript.

## Conclusion

Both novelty claims were confirmed directly in Scopus and in Web of
Science on 22 November 2026: no prior study combines graph-neural-network
active-learning screening with DFT validation for calcium-phosphate
bone scaffolds, and no prior study uses a literature-verified
biocompatibility gate with a backtested selection rule for this material
class.
