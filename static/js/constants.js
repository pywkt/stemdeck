import { t, TRANSLATIONS } from "./i18n.js";

// Fallback list used before /api/config responds. Kept in sync with
// STEM_NAMES in app/core/config.py — the API is the canonical source.
export let STEM_NAMES = ["vocals", "drums", "bass", "guitar", "piano", "other"];
export let TRACK_NAMES = ["original", ...STEM_NAMES];

// Stems from the on-demand lead/backing vocal split (#275, EXTRA_STEM_NAMES
// in app/core/config.py). Not folded into STEM_NAMES/TRACK_NAMES -- most jobs
// never run this split -- but the player/mixer render them as real lanes,
// swapped in for "vocals", via effectiveStemOrder() below.
export let EXTRA_STEM_NAMES = ["lead_vocals", "backing_vocals"];

// The stems the CURRENTLY selected separation model can produce. STEM_NAMES
// stays the canonical superset that colours, labels and ordering are keyed by;
// this is the subset the import UI offers, so choosing a vocal model (which
// makes only vocals + a full instrumental) stops offering drums/bass/guitar/
// piano chips for stems it cannot make. Starts as the full set until
// /api/config answers.
export let ACTIVE_STEMS = [...STEM_NAMES];

export async function syncStemNamesFromAPI() {
  try {
    const res = await fetch("/api/config", { cache: "no-store" });
    if (!res.ok) return;
    const data = await res.json();
    if (Array.isArray(data.stem_names) && data.stem_names.length > 0) {
      STEM_NAMES = data.stem_names;
      TRACK_NAMES = ["original", ...STEM_NAMES];
    }
    if (Array.isArray(data.extra_stem_names) && data.extra_stem_names.length > 0) {
      EXTRA_STEM_NAMES = data.extra_stem_names;
    }
    if (Array.isArray(data.active_stems) && data.active_stems.length > 0) {
      // Filtered through STEM_NAMES rather than used as-is, so the chips keep
      // canonical order whatever order the registry happens to list them in.
      ACTIVE_STEMS = STEM_NAMES.filter((n) => data.active_stems.includes(n));
    }
  } catch (e) {
    console.warn("[constants] failed to sync stem names from API:", e);
  }
}

// The lane order for a specific job: STEM_NAMES with "vocals" replaced by
// lead_vocals + backing_vocals when a job's on-demand split (#275) has
// produced both. `presentNames` is the Set of stem names the job actually
// has (typically from job.stems). Order matters -- callers use this both to
// decide what to render and in what sequence (waveform stacking, mixer rows).
export function effectiveStemOrder(presentNames) {
  const splitDone = presentNames.has("lead_vocals") && presentNames.has("backing_vocals");
  return STEM_NAMES.flatMap((n) => (n === "vocals" && splitDone ? EXTRA_STEM_NAMES : [n]));
}

// A Proxy, not a plain object, so every lookup resolves through the CURRENT
// language live -- a plain object would freeze these labels in whatever
// language was active the moment this module first loaded. Callers keep their
// existing `STEM_DISPLAY[name] || name` fallback pattern unchanged: an
// unrecognized name (not one of ours) still yields undefined, same as a plain
// object would, rather than the i18n engine's own missing-key fallback
// (which returns the raw dictionary key string -- wrong here).
export const STEM_DISPLAY = new Proxy({}, {
  get(_target, prop) {
    if (typeof prop !== "string") return undefined;
    const key = `stem.${prop}`;
    return key in TRANSLATIONS.en ? t(key) : undefined;
  },
});

// FL Studio-style channel palette: saturated but slightly dusty, designed
// to read well on a dark background.
export const STEM_COLORS = {
  vocals: "#e85f6f",
  drums: "#e89048",
  bass: "#e8b848",
  guitar: "#88d878",
  piano: "#b88fe8",
  other: "#88a8c8",
  original: "#a8b0bd",
  lead_vocals: "#e8748a",
  backing_vocals: "#c98fe0",
};

export const PROGRESS_COLOR = "#3a3a3a";

export const LOOP_DEFAULT_START_FRAC = 0.25;
export const LOOP_DEFAULT_END_FRAC = 0.5;

export const LANE_VOLUME_MAX = 2;