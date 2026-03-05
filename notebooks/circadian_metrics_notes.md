# Potential additional circadian metrics

## Rhythm strength & regularity

| Metric | Description | Status |
|--------|-------------|--------|
| **IS** : Inter-daily Stability | Consistency of the rhythm across days | implemented |
| **IV** : Intra-daily Variability | Fragmentation of the rest-activity rhythm within a day | implemented |
| **Tau (τ)** : Period | Dominant period of the rhythm, estimated via periodogram | todo |

For τ, use a periodogram (e.g. Lomb-Scargle or chi-square). See the `chronobiology` Python package (see [repo](https://github.com/ruslands/chronobiology) and [docs](https://chronobiology.readthedocs.io/en/latest/analyzing_data.html#Periodogram)).

---

## Phase & timing

| Metric | Description | Notes |
|--------|-------------|-------|
| **Activity onset** | Time at which sustained activity begins each day; most stable phase marker, conventionally CT12 | |
| **Alpha (α)** | Duration of the active phase within a circadian cycle | |
| **Rho (ρ)** | Duration of the rest phase within a circadian cycle | |
| **Phase angle of entrainment (ψ)** | Lag between the light-off signal and activity onset under LD cycles | Check both dusk and full-lights-off transitions |

---

## Light-dark preference

| Metric | Description | Status |
|--------|-------------|--------|
| **DI** : Diurnality Index | `(A_light - A_dark) / (A_light + A_dark)`; nocturnal animals expected to be negative | implemented |

Additional visualisation: sort individuals by their correlation to the LD square-wave pattern (can be done from the actogram or activity time series).

See also: *Diurnality is consistently different between individuals and decreases with disease or stressful events in dairy calves* ([Sci. Reports 2025](https://www.nature.com/articles/s41598-025-09983-z#Fig4)).

---

## Fragmentation & bout structure

| Metric | Description | Notes |
|--------|-------------|-------|
| **Bout analysis** | Number, duration, and inter-bout interval of activity bouts | Convention: 15-min bins; 5-min bins may be better here |
| **Sleep continuity** | Number of naps / sleep bouts; longest continuous rest period | |

---

## Reference implementations

| Tool | Language | Notes |
|------|----------|-------|
| [`digiRhythm`](https://cran.r-project.org/web/packages/digiRhythm/vignettes/Actgram_diurnality_avg_activity.html) | R | Actogram, average daily activity, diurnality index |
| [`chronobiology`](https://chronobiology.readthedocs.io/en/latest/chronobiology.html#chronobiology.chronobiology.CycleAnalyzer) | Python | Periodogram, cycle analysis |
| [`ActogramJ`](https://bene51.github.io/ActogramJ/download.html) | Java | Broad set of metrics; last updated 2022 |
| [Tackenberg et al. 2019](https://journals.sagepub.com/doi/10.1177/0748730419862474) | R | Non-parametric circadian rhythm analysis |
