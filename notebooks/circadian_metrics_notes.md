# Potential additional circadian metrics

## Rhythm strength & regularity

| Metric | Description | Status |
|--------|-------------|--------|
| **IS** : Inter-daily Stability | Consistency of the rhythm across days | implemented |
| **IV** : Intra-daily Variability | Fragmentation of the rest-activity rhythm within a day | implemented |
| **RA** : Relative Amplitude | Ratio of activity in the most active 10-hour window to the least active 5-hour window; reduced when circadian rhythms are disrupted | implemented |
| **DFC** : Degree of Functional Coupling | Proportion of spectral power (Lomb-Scargle) concentrated at 24-hour harmonics (24h, 12h, 8h, ...); expressed as % | todo |
| **HP** : Harmonic Power | Total spectral power at 24-hour harmonic frequencies; companion metric to DFC | todo |
| **Tau (τ)** : Period | Dominant period estimated from a periodogram | todo |

**DFC and HP** are computed over a rolling window (7 days standard) from the Lomb-Scargle periodogram. High values (near 100%) indicate strong circadian organisation; low values suggest poor synchronisation to the 24-hour cycle. Significance testing uses Baluev's method. Recommended bin size: 15–30 min. See [`digiRhythm`](https://cran.r-project.org/web/packages/digiRhythm/vignettes/DFC_and_HP_and_changing_plots.html).

**RA, IS, IV** are all available in the `chronobiology` Python package ([`CycleAnalyzer`](https://chronobiology.readthedocs.io/en/latest/chronobiology.html#chronobiology.chronobiology.CycleAnalyzer)).

**Tau** can be estimated with Fourier, chi-square (Enright), or Lomb-Scargle periodogram. See `chronobiology` ([docs](https://chronobiology.readthedocs.io/en/latest/analyzing_data.html#Periodogram)) or ActogramJ for a GUI-based approach.

---

## Phase & timing

| Metric | Description | Notes |
|--------|-------------|-------|
| **Activity onset** | Time at which sustained activity begins each day; most stable phase marker, conventionally CT12 | `chronobiology`: `activity_onset()` uses kernel convolution |
| **Alpha (α)** | Duration of the active phase within a circadian cycle | Derivable from onset + offset |
| **Rho (ρ)** | Duration of the rest phase within a circadian cycle | 24h − α |
| **Phase angle of entrainment (ψ)** | Lag between the light-off signal and activity onset under LD cycles | Check both dusk and full-lights-off transitions |
| **Acrophase** | Time of peak activity in the average daily profile | ActogramJ: regression line can be fit across days |

Activity onsets/offsets and acrophases can be computed in ActogramJ with optional regression lines to track drift across days.

---

## Light-dark preference

| Metric | Description | Status |
|--------|-------------|--------|
| **DI** : Diurnality Index | `(A_light - A_dark) / (A_light + A_dark)`; nocturnal animals expected to be negative | implemented |
| **Light-phase activity** | Absolute activity during the normally inactive (light) phase; flags entrainment defects and misaligned feeding | todo |

Additional visualisation: sort individuals by their correlation to the LD square-wave pattern (can be done from the actogram or activity time series).

`chronobiology`: `light_activity()` method. `digiRhythm`: `diurnality()` with configurable day/night windows.

See also: *Diurnality is consistently different between individuals and decreases with disease or stressful events in dairy calves* ([Sci. Reports 2025](https://www.nature.com/articles/s41598-025-09983-z#Fig4)).

---

## Fragmentation & bout structure

| Metric | Description | Notes |
|--------|-------------|-------|
| **Bout count & duration** | Number of activity bouts per day, mean bout duration, inter-bout interval | Circadian disruption → more bouts, shorter duration |
| **Sleep continuity** | Longest continuous rest period; number of rest bouts | |

`chronobiology`: `activity_bouts()` and `daily_bouts()` methods. Parameters: `max_gap` (gap to merge bouts), `min_duration` (minimum bout length), `min_activity` (activity threshold). Convention: 15-min bins; 5-min may be better for mice.

---

## Reference implementations

| Tool | Language | Relevant metrics |
|------|----------|-----------------|
| [`chronobiology`](https://chronobiology.readthedocs.io/en/latest/chronobiology.html#chronobiology.chronobiology.CycleAnalyzer) | Python | IS, IV, RA, τ (periodogram), activity onset, light-phase activity, bout analysis |
| [`digiRhythm`](https://cran.r-project.org/web/packages/digiRhythm/) | R | DFC, HP, DI, actogram, average daily activity |
| [`ActogramJ`](https://bene51.github.io/ActogramJ/download.html) | Java (GUI) | τ (Fourier / chi-square / Lomb-Scargle), acrophase, onset/offset regression; last updated 2022 |
| [Tackenberg et al. 2019](https://journals.sagepub.com/doi/10.1177/0748730419862474) | R | Non-parametric circadian rhythm analysis (IS, IV, RA, M10, L5) |
