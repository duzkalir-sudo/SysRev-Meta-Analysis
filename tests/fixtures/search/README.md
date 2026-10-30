# Synthetic search exports (test fixtures only)

All entries are **invented**. DOIs use the 10.5555 test prefix, NCT numbers and accession
numbers are fictitious, and none of these records describe real studies. The files imitate
the general shape of RIS, BibTeX and CSV exports; they are not copies of any vendor's
current export format.

Deliberate overlaps (used by the deduplication and PRISMA tests):

* `embase_synthetic.ris` #1, `scopus_synthetic.bib` `ref1` and PubMed PMID 90000001
  (`../pubmed_efetch_synthetic.xml`) are the same report (shared DOI 10.5555/synth.0001).
* `embase_synthetic.ris` #3 and `scopus_synthetic.bib` `ref2` share DOI 10.5555/synth.0002
  but have unrelated titles: an ambiguous match that must go to a human, not be merged.
* `central_synthetic.ris` #1 repeats PubMed PMID 90000002 without identifiers (title match).
