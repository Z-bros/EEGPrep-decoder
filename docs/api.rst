API reference
=============

Every public function and class, grouped by subpackage. Inputs are MNE
objects (or NumPy arrays through :func:`eeg_decoder.io.to_raw`), signals are
in volts, and inputs are never modified. The assumptions behind each
function are in the :doc:`module-notes`.

Top level
---------

.. currentmodule:: eeg_decoder

.. autosummary::
   :toctree: generated/
   :nosignatures:

   EEGConfig

Input
-----

``eeg_decoder.io``

.. currentmodule:: eeg_decoder.io

.. autosummary::
   :toctree: generated/
   :nosignatures:

   to_raw

Preprocessing
-------------

``eeg_decoder.preprocessing``

.. currentmodule:: eeg_decoder.preprocessing

.. autosummary::
   :toctree: generated/
   :nosignatures:

   bandpass
   notch
   rereference
   resample
   PreprocessingConfig
   EEGPipeline

Epoching
--------

``eeg_decoder.epoching``

.. currentmodule:: eeg_decoder.epoching

.. autosummary::
   :toctree: generated/
   :nosignatures:

   make_epochs
   baseline_correct

Artifacts
---------

``eeg_decoder.artifacts``

.. currentmodule:: eeg_decoder.artifacts

.. autosummary::
   :toctree: generated/
   :nosignatures:

   assess_channels
   ChannelAssessment
   mark_bad_channels
   interpolate_bad_channels
   annotate_artifacts
   add_annotations
   reject_epochs
   fit_ica
   find_artifact_components
   ComponentReport
   apply_ica

Signal quality
--------------

``eeg_decoder.quality``

.. currentmodule:: eeg_decoder.quality

.. autosummary::
   :toctree: generated/
   :nosignatures:

   recording_quality
   RecordingQuality
   epochs_quality
   EpochsQuality
   plot_psd_comparison
   plot_channel_quality
   plot_rejection_summary

Dataset preparation and export
------------------------------

``eeg_decoder.dataset``

.. currentmodule:: eeg_decoder.dataset

.. autosummary::
   :toctree: generated/
   :nosignatures:

   check_compatible
   match_channels
   prepare_recordings
   combine_epochs
   EpochDataset
   save_npz
   load_npz
   save_epochs_fif
   load_epochs_fif

Optional features
-----------------

``eeg_decoder.features``

Not used by any of the modules above.

.. currentmodule:: eeg_decoder.features

.. autosummary::
   :toctree: generated/
   :nosignatures:

   FeatureArray
   welch_psd
   band_power
   time_stats
   hjorth
   morlet_tfr

``CLASSIC_BANDS`` is a plain ``dict`` of example bands (delta 1-4, theta 4-8,
alpha 8-13, beta 13-30, gamma 30-45 Hz) to pass explicitly.
