# Signal quality

Descriptive metrics (never pass/fail verdicts) and plots, to compare
recordings and document data quality before and after preprocessing.

```python
import json
from pathlib import Path
from eeg_decoder.quality import (
    recording_quality, epochs_quality,
    plot_psd_comparison, plot_channel_quality, plot_rejection_summary,
)

q = recording_quality(raw, line_freq=50)     # line frequency is never assumed
print(q.summary())                           # bad channels, BAD time, amplitude, line noise
Path("quality.json").write_text(json.dumps(q.to_dict(), indent=2))  # machine-readable

eq = epochs_quality(epochs)                  # kept / dropped, and why (from drop_log)

fig = plot_psd_comparison(raw_original, raw_clean)   # mean PSD of good channels
fig.savefig("psd.png")                               # figures are returned, never shown or saved
```

- **`recording_quality`**: bad EEG channels and their fraction; time covered
  by `BAD*` annotations (union, clipped to the recording, exactly as epoching
  sees it) and per annotation description; per-channel robust SD and
  peak-to-peak in volts; and, with `line_freq`, a line-noise ratio: power
  within ±0.5 Hz of the line frequency vs the median power 2-5 Hz away, in
  dB (about 0 dB means no line noise, negative after a notch). Channel
  metrics use only data outside `BAD*` annotations.
- **`epochs_quality`**: events kept and dropped, drop reasons (channel names
  for threshold rejection, annotation descriptions, MNE's `NO_DATA` etc.),
  kept epochs per condition. Events MNE marks `IGNORED` (code not in
  `event_id`) are not counted as candidates. MNE does not keep the codes of
  dropped epochs, so per-condition counts cover kept epochs only.
- **Plots** use matplotlib's object-oriented `Figure`, not `pyplot`: no
  window, no global state, safe on headless servers. Colours come from a
  colour-vision-deficiency-checked palette, and bad channels are also
  hatched, so meaning never depends on colour alone.
