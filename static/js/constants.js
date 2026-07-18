// STEM_NAMES is the canonical SUPERSET (all colours/labels/ordering derive from
// it). ACTIVE_STEMS is the subset the currently selected model produces — the
// import UI (stem-choice chips) offers only these, so a vocal model shows just
// vocals+other. Both start as the 6-stem fallback until /api/config responds.
export let STEM_NAMES = ["vocals", "drums", "bass", "guitar", "piano", "other"];
export let TRACK_NAMES = ["original", ...STEM_NAMES];
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
    if (Array.isArray(data.active_stems) && data.active_stems.length > 0) {
      // Keep canonical order for the chips regardless of registry order.
      ACTIVE_STEMS = STEM_NAMES.filter((n) => data.active_stems.includes(n));
    }
  } catch (e) {
    console.warn("[constants] failed to sync stem names from API:", e);
  }
}

export const STEM_DISPLAY = {
  vocals: "Vocals",
  drums: "Drums",
  bass: "Bass",
  guitar: "Guitar",
  piano: "Piano",
  other: "Other",
  original: "Original",
};

// The other-instrument stems a multi-stem model separates out. When NONE of
// them is present in a job (a vocal model's vocals+other), "other" IS the full
// instrumental backing track, so we label it "Instrumental" instead of "Other".
const _INSTRUMENT_STEMS = ["drums", "bass", "guitar", "piano"];

// Display label for a stem, given the job's full stem-name set. Special-cases
// "other" -> "Instrumental" for 2-stem vocal jobs; otherwise STEM_DISPLAY.
export function stemLabel(name, jobStemNames = null) {
  if (
    name === "other" &&
    Array.isArray(jobStemNames) &&
    !_INSTRUMENT_STEMS.some((s) => jobStemNames.includes(s))
  ) {
    return "Instrumental";
  }
  return STEM_DISPLAY[name] || name;
}

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
};

export const PROGRESS_COLOR = "#3a3a3a";

export const LOOP_DEFAULT_START_FRAC = 0.25;
export const LOOP_DEFAULT_END_FRAC = 0.5;

export const LANE_VOLUME_MAX = 2;